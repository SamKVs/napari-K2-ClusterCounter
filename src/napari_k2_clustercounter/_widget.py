"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QWidget, QInputDialog, QFileDialog
from qtpy.QtGui import QPixmap
from qtpy import uic
import numpy as np
from skimage.morphology import remove_small_objects, disk, binary_dilation
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import measure
from scipy import ndimage as ndi
from .DrawLine import draw_path
from .ClickLabel import ClickLabel
from .AreaToLength import area_to_length
import sqlite3
import pandas as pd
import os
import napari
from napari_plugin_engine import napari_hook_implementation


class ArrayShapeIncompatible(Exception):
    """Raised when the input value is too small"""
    pass


# Define the main widget window for the plugin
class ClusterCounter(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.UI_FILE = str(f"{os.path.dirname(os.path.abspath(__file__))}/static/form.ui")  # path to .ui file
        uic.loadUi(self.UI_FILE, self)

        # Functional UI dynamics
        self.dict_organize = {
            "ManualROI": [self.roiTypeLabel,
                          self.roiTypeComboBox,
                          self.genRoiAreaButton,
                          self.maxLine,
                          self.maxLineLabel,
                          self.genRoiLineButton,
                          self.clipLineButton],
            "ROIChannel": [self.roiLabel,
                           self.roiComboBox,
                           self.roiThreshLabel,
                           self.roiThreshSlider,
                           self.roiBlur,
                           self.roiBlurLabel,
                           self.roiDilations,
                           self.roiDilationsLabel,
                           self.genRoiButton],
            "ManualROIArea": [self.genRoiAreaButton],
            "ManualROILine": [self.genRoiLineButton,
                              self.maxLine,
                              self.maxLineLabel,
                              self.clipLineButton],
            "LocalMaxPeak": [self.lmpThreshLabel,
                             self.lmpThreshSlider,
                             self.lmpThreshGenerate,
                             self.lmpMinDistLabel,
                             self.lmpMinDistSpinBox]
        }

        self.activeOption = "Manual"
        self.changevis(self.dict_organize["ROIChannel"], False)
        self.typeImgMchanged()
        self.TempDb = None

        # Udate layer list
        self.updatelayer()
        self.viewer.layers.events.removed.connect(self.updatelayer)
        self.viewer.layers.events.inserted.connect(self.updatelayer)
        self.viewer.layers.events.changed.connect(self.updatelayer)

        # Update basin definition
        self.updatebasins()
        self.basinComboBox.currentTextChanged.connect(self.updatebasins)
        self.lmpThreshGenerate.clicked.connect(self.LocalMaxPeakShow)

        # Update scale
        self.clusterComboBox.currentTextChanged.connect(self.updatescale)

        # Dynamic UI Elements
        self.manualButton.clicked.connect(self.typeImgM)
        self.roiButton.clicked.connect(self.typeImgR)
        self.roiTypeComboBox.currentTextChanged.connect(self.typeImgMchanged)

        # Generate cluster threshold
        self.threshButton.clicked.connect(self.ClusterThresholdShow)

        # Generate ROI layer
        self.genRoiAreaButton.clicked.connect(self.genROIArea)
        self.genRoiLineButton.clicked.connect(self.genROILine)
        self.genRoiButton.clicked.connect(self.genROI)
        self.clipLineButton.clicked.connect(self.clipLine)

        # Watershedding Analysis
        self.analyzeButton.clicked.connect(self.WatershedPreview)
        self.analyzeAddButton.clicked.connect(self.WatershedAdd)

        # Edit List
        self.renameButton.clicked.connect(self.renameList)
        self.deleteButton.clicked.connect(self.deleteList)

        # Save Analysis
        self.saveResultsButton.clicked.connect(self.saveResults)

        # Make sure sqLite database is empty
        if os.path.exists(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite"):
            os.remove(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
        con = sqlite3.connect(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
        con.close()

    # UTILITY FUNCTIONS
    def updatescale(self):
        try:
            scale = self.viewer.layers[self.clusterComboBox.currentText()].scale
            if len(scale) == 2 and scale[0] == scale[1]:
                self.scaleBox.setValue(scale[0])
            else:
                self.scaleBox.setValue(scale[0])
        except Exception:
            pass

    def getShape(self, array):
        if array == "cluster":
            return np.shape(self.viewer.layers[self.clusterComboBox.currentText()].data)
        elif array == "roi":
            return np.shape(self.viewer.layers[self.roiComboBox.currentText()].data)

    def changevis(self, list, bool):
        for x in list:
            x.setVisible(bool)

    def typeImgMchanged(self):
        if self.roiTypeComboBox.currentText() == "Area":
            self.typeImgMArea()
        elif self.roiTypeComboBox.currentText() == "Line":
            self.typeImgMLine()

    def ClusterThreshold(self, input, thresh, minIslandSize, connectivity):

        mask = np.copy(input)
        ### MANUAL THRESHOLDING ###
        thresh = (thresh / 1000) * np.max(input)
        print(thresh)
        mask[mask <= thresh] = 0
        mask[mask != 0] = 1
        print(connectivity)

        mask = remove_small_objects(mask.astype(bool), minIslandSize,
                                    connectivity=2 if connectivity is True else 1)

        return mask

    def RoiThreshold(self, input, threshold, blur, dilations):
        ### BLUR ###
        mask = gaussian(input, blur)
        ### MANUAL THRESHOLDING ###
        thresh = (threshold / 1000) * np.max(mask)
        print(thresh)
        mask[mask <= thresh] = 0
        mask[mask != 0] = 1

        mask = np.squeeze(mask)

        ### DILATIONS ###
        mask = binary_dilation(mask.astype(bool), disk(int(dilations)))

        return mask

    def clipLinePath(self, path):
        print("Clipping line path", path)
        maxl = self.maxLine.value()
        dif = np.diff(path, axis=0)
        distance = np.sqrt(np.sum(np.square(np.abs(dif)), axis=1))
        print("dif", dif)
        print("distance", distance)
        cumsum = np.cumsum(distance)
        print("cumsum", cumsum)

        if cumsum[-1] < maxl:
            return path
        P1 = len(cumsum[cumsum < maxl])
        P2 = P1 + 1
        left = maxl - cumsum[cumsum < maxl][-1] if len(cumsum[cumsum < maxl]) != 0 else maxl
        relativeleft = left / distance[P1]
        print(P1, P2, left, relativeleft)
        return np.concatenate((path[:P2], np.array([path[P1] + relativeleft * dif[P1]])), axis=0)

    def LocalMaxPeak(self, array, mask, thresh, minDist):
        peaks = peak_local_max(array,
                               min_distance=minDist,
                               threshold_rel=thresh / 1000,
                               labels=mask,
                               exclude_border=False)
        return peaks

    def Watershed(self, mask, coords):
        peaks = np.zeros(mask.shape, dtype=bool)
        if len(coords) != 0:
            peaks[tuple(coords.T)] = True
        markers, _ = ndi.label(peaks)
        shedded = watershed(mask,
                            markers=markers,
                            mask=mask)
        return shedded

    def getlength(self, poly):
        dif = np.diff(poly, axis=0)
        distance = np.sqrt(np.sum(np.square(np.abs(dif)), axis=1))
        return np.sum(distance)

    def ConvertLinesToBinary(self, polys, width):
        field = self.getShape("cluster")
        print(field)
        canvases = []
        for index, poly in enumerate(polys):
            canvas = np.zeros(field, dtype=bool)
            canvas = draw_path(poly, width[index], canvas)

            #Background
            if self.lineBackSpinBox.value() != 0:
                canvas_back = np.zeros(field, dtype=bool)
                canvas_back = draw_path(poly, width[index] * self.lineBackSpinBox.value(), canvas_back)
                line_back = canvas_back * np.invert(self.viewer.layers[self.clusterComboBox.currentText() + '_mask'].data.astype(bool))
                self.viewer.add_labels(line_back, name=self.clusterComboBox.currentText() + '_line_back')
                intensity_back = np.mean(self.viewer.layers[self.clusterComboBox.currentText()].data[line_back])

            if self.TempDb is not None:
                self.TempDb["Line"]["length"].append(self.getlength(poly))
                self.TempDb["Line"]["width"].append(width[index])
                self.TempDb["Line"]["area"].append(np.sum(canvas))
                self.TempDb["Line"]["bg_width"].append(width[index] * self.lineBackSpinBox.value())
                self.TempDb["Line"]["bg_area"].append(np.sum(line_back))
                self.TempDb["Line"]["bg_intensity"].append(intensity_back)
            canvases.append(canvas)
        return canvases

    def UpdateListWidget(self, listNames):
        self.listList.clear()
        self.listList.addItems(listNames)

    def FetchFromSQL(self, c):
        return list(sum(c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall(), ()))

    def renameList(self):
        if len(self.listList.selectedItems()) == 1:
            new, ok = QInputDialog.getText(self, 'Rename List', 'Enter a new name:',
                                            text=self.listList.currentItem().text())
            print(self.listList.currentItem().text())
            if ok:
                if new == self.listList.currentItem().text():
                    pass
                else:
                    con = sqlite3.connect(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
                    c = con.cursor()
                    c.execute(f"ALTER TABLE '{self.listList.currentItem().text()}' RENAME TO '{new}' ")
                    self.UpdateListWidget(self.FetchFromSQL(c))
                    self.listList.setCurrentRow(self.listList.count() - 1)
                    con.commit()
                    con.close()
        else:
            print("No or Too many items selected")

    # UI FUNCTIONS
    def typeImgM(self):
        self.manualButton.setPixmap(QPixmap(str(f"{os.path.dirname(os.path.abspath(__file__))}/static/MAN_T.png")))
        self.roiButton.setPixmap(QPixmap(str(f"{os.path.dirname(os.path.abspath(__file__))}/static/ROI_F.png")))
        self.changevis(self.dict_organize["ManualROI"], True)
        self.changevis(self.dict_organize["ROIChannel"], False)
        self.typeImgMchanged()
        self.activeOption = "Manual"

    def typeImgR(self):
        self.manualButton.setPixmap(QPixmap(str(f"{os.path.dirname(os.path.abspath(__file__))}/static/MAN_F.png")))
        self.roiButton.setPixmap(QPixmap(str(f"{os.path.dirname(os.path.abspath(__file__))}/static/ROI_T.png")))
        self.changevis(self.dict_organize["ROIChannel"], True)
        self.changevis(self.dict_organize["ManualROI"], False)
        self.activeOption = "ROI"

    def typeImgMArea(self):
        self.changevis(self.dict_organize["ManualROIArea"], True)
        self.changevis(self.dict_organize["ManualROILine"], False)

    def typeImgMLine(self):
        self.changevis(self.dict_organize["ManualROIArea"], False)
        self.changevis(self.dict_organize["ManualROILine"], True)

    def updatelayer(self):
        one = self.clusterComboBox.currentText()
        two = self.roiComboBox.currentText()
        three = self.addChannelComboBox.currentText()

        self.clusterComboBox.clear()
        self.roiComboBox.clear()
        self.addChannelComboBox.clear()

        self.addChannelComboBox.addItem("None")

        for i in self.viewer.layers:
            self.clusterComboBox.addItem(str(i))
            self.roiComboBox.addItem(str(i))
            self.addChannelComboBox.addItem(str(i))


        if one in self.viewer.layers:
            self.clusterComboBox.setCurrentText(one)
        if two in self.viewer.layers:
            self.roiComboBox.setCurrentText(two)
        if three in self.viewer.layers:
            self.addChannelComboBox.setCurrentText(three)

    def updatebasins(self):
        if self.basinComboBox.currentText() == "Distance tranformation":
            self.changevis(self.dict_organize["LocalMaxPeak"], False)
        else:
            self.changevis(self.dict_organize["LocalMaxPeak"], True)

    def clipLine(self):
        pathlist = self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].data

        for index, path in enumerate(pathlist):
            npath = self.clipLinePath(path)
            pathlist[index] = npath

        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].data = pathlist

    def deleteList(self):
        con = sqlite3.connect(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
        c = con.cursor()
        for i in self.listList.selectedItems():
            command = f"""DROP TABLE '{i.text()}'"""
            c.execute(command)
        self.UpdateListWidget(self.FetchFromSQL(c))
        con.commit()
        con.close()
        pass

    # THRESHOLDING FUNCTIONS
    def ClusterThresholdShow(self):
        if (str(self.clusterComboBox.currentText()) + "_mask") in self.viewer.layers:
            self.viewer.layers.remove(str(self.clusterComboBox.currentText()) + "_mask")
        thresh = self.threshSlider.value()
        minIslandSize = self.minIslandSize.value()
        allowDiogonalConnectivity = self.connectivityCheckbox.isChecked()
        mask = self.ClusterThreshold(self.viewer.layers[self.clusterComboBox.currentText()].data,
                                     thresh,
                                     minIslandSize,
                                     allowDiogonalConnectivity)
        self.viewer.add_labels(mask,
                               name=(str(self.clusterComboBox.currentText()) + "_mask"),
                               color={1: "purple"},
                               opacity=0.30)
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].scale = self.viewer.layers[
            self.clusterComboBox.currentText()].scale

    # ROI FUNCTIONS
    def genROIArea(self):
        # TODO: Add labels layer
        # Delete old ROI if it exists
        if (str(self.clusterComboBox.currentText()) + "_ROI") in self.viewer.layers:
            self.viewer.layers.remove(str(self.clusterComboBox.currentText()) + "_ROI")

        self.viewer.add_labels(np.zeros(self.getShape("cluster"), dtype=np.uint8),
                               name=(str(self.clusterComboBox.currentText()) + "_ROI"),
                               color={1: "blue"},
                               opacity=0.30)
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].mode = "paint"
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].brush_size = 70
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].scale = self.viewer.layers[
            self.clusterComboBox.currentText()].scale

        pass

    def genROILine(self):
        # Delete old ROI if it exists
        if (str(self.clusterComboBox.currentText()) + "_ROI") in self.viewer.layers:
            self.viewer.layers.remove(str(self.clusterComboBox.currentText()) + "_ROI")
        self.viewer.add_shapes(data=None,
                               name=(str(self.clusterComboBox.currentText()) + "_ROI"),
                               edge_color="blue",
                               edge_width=30,
                               opacity=0.30)
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].mode = "ADD_PATH"
        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].scale = self.viewer.layers[
            self.clusterComboBox.currentText()].scale

        pass

    def genROI(self):
        # Delete old ROI if it exists
        if (str(self.roiComboBox.currentText()) + "_ROI") in self.viewer.layers:
            self.viewer.layers.remove(str(self.roiComboBox.currentText()) + "_ROI")

        ROImask = self.RoiThreshold(self.viewer.layers[self.roiComboBox.currentText()].data,
                                    self.roiThreshSlider.value(),
                                    self.roiBlur.value(),
                                    self.roiDilations.value())

        self.viewer.add_labels(ROImask,
                               name=(str(self.roiComboBox.currentText()) + "_ROI"),
                               color={1: "blue"},
                               opacity=0.30)

        self.viewer.layers[str(self.roiComboBox.currentText()) + "_ROI"].mode = "erase"
        self.viewer.layers[str(self.roiComboBox.currentText()) + "_ROI"].brush_size = 70
        self.viewer.layers[str(self.roiComboBox.currentText()) + "_ROI"].scale = self.viewer.layers[
            self.roiComboBox.currentText()].scale

        pass

    # Basin functions
    def LocalMaxPeakShow(self, final=False):
        # Remove existing _LocalMaxPeak layers
        string_list = [layer.name for layer in self.viewer.layers]
        for i in string_list:
            if (str(self.clusterComboBox.currentText()) + "_LocalMaxPeak") in i:
                self.viewer.layers.remove(i)

        # Get parameters
        minDist = self.lmpMinDistSpinBox.value()
        thresh = self.lmpThreshSlider.value()

        # Generate _mask if not available
        if not (str(self.clusterComboBox.currentText()) + "_mask") in self.viewer.layers:
            self.ClusterThresholdShow()

        ROIs = []
        # Check if ROI layers exist and should be applied
        print(self.activeOption)
        if self.activeOption == "Manual":
            if (str(self.clusterComboBox.currentText()) + "_ROI") in self.viewer.layers:
                if self.roiTypeComboBox.currentText() == "Area":
                    if final:
                        self.TempDb["Area"] = np.sum(
                            self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].data)
                    ROIs.append(
                        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].data * self.viewer.layers[
                            str(self.clusterComboBox.currentText()) + "_ROI"].data)
                elif self.roiTypeComboBox.currentText() == "Line":
                    if final:
                        self.TempDb["Line"] = {"length": [], "width": [], "area": [], "bg_width": [], "bg_area": [], "bg_intensity": []}
                    LineROIs = self.ConvertLinesToBinary(
                        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].data,
                        self.viewer.layers[str(self.clusterComboBox.currentText()) + "_ROI"].edge_width)

                    for canvas in LineROIs:
                        ROIs.append(self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].data * canvas)

            else:
                # TODO: Pass warning message that no manual mask is available
                pass
        if self.activeOption == "ROI":
            if final:
                self.TempDb["ROI"] = {"area": np.sum(self.viewer.layers[str(self.roiComboBox.currentText()) + "_ROI"].data),
                                      "length": area_to_length(self.viewer.layers[str(self.roiComboBox.currentText()) + "_ROI"].data)}

            if (str(self.roiComboBox.currentText()) + "_ROI") in self.viewer.layers:
                ROIs.append(
                    self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].data * self.viewer.layers[
                        str(self.roiComboBox.currentText()) + "_ROI"].data)

            else:
                # TODO: Pass warning message that no auto mask is available
                pass

        if ROIs == []:
            ROIs.append(self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].data)

        for index, ROI in enumerate(ROIs):
            ROI = ROI.astype(bool)
            peaks = self.LocalMaxPeak(self.viewer.layers[self.clusterComboBox.currentText()].data,
                                      ROI,
                                      thresh,
                                      minDist)

            self.viewer.add_points(peaks,
                                   name=(str(self.clusterComboBox.currentText()) + "_LocalMaxPeak_" + str(index)),
                                   face_color="blue",
                                   size=2)
            self.viewer.layers[str(self.clusterComboBox.currentText()) + "_LocalMaxPeak_" + str(index)].scale = \
                self.viewer.layers[self.clusterComboBox.currentText()].scale

    def WatershedPreview(self, final=False):
        # Remove any existing watershed layer
        for i in [layer.name for layer in self.viewer.layers if "Watershed" in layer.name]:
            self.viewer.layers.remove(i)

        # Make sure mask and local max peak layers exist
        if not (str(self.clusterComboBox.currentText()) + "_LocalMaxPeak") in self.viewer.layers:
            self.LocalMaxPeakShow(final=final)
        if not (str(self.clusterComboBox.currentText()) + "_mask") in self.viewer.layers:
            self.ClusterThresholdShow()

        mask = self.viewer.layers[str(self.clusterComboBox.currentText()) + "_mask"].data

        string_list = [layer.name for layer in self.viewer.layers if "LocalMaxPeak" in layer.name]
        for i in string_list:
            index = i.split("_")[-1]

            shedded = self.Watershed(mask, self.viewer.layers[i].data)

            self.viewer.add_labels(shedded,
                                   name=(str(self.clusterComboBox.currentText()) + "_Watershed_" + str(index)),
                                   opacity=0.80)
            self.viewer.layers[str(self.clusterComboBox.currentText()) + "_Watershed_" + str(index)].scale = \
                self.viewer.layers[self.clusterComboBox.currentText()].scale

            pass

    def WatershedAdd(self):
        self.TempDb = {}

        def findSmallestIntNotInList(number_list):
            # Infinite while-loop until first number is found
            i = 0
            while True:
                i += 1
                if i not in number_list:
                    return i

        # Open Metadata database
        # Check if watershed layer exists
        if not (str(self.clusterComboBox.currentText()) + "_Watershed") in self.viewer.layers:
            self.WatershedPreview(final=True)

        print(self.TempDb)

        print([layer.name for layer in self.viewer.layers if "Watershed" in layer.name])
        for index, i in enumerate([layer.name for layer in self.viewer.layers if "Watershed" in layer.name]):
            props = measure.regionprops_table(self.viewer.layers[i].data,
                                              self.viewer.layers[self.clusterComboBox.currentText()].data,
                                              properties=['label',
                                                          'area',
                                                          'centroid',
                                                          'mean_intensity',
                                                          'max_intensity',
                                                          'min_intensity',
                                                          'feret_diameter_max',
                                                          'eccentricity',
                                                          'orientation',
                                                          'perimeter'])
            # Properties to pd dataframe
            df = pd.DataFrame(props)

            # If dataframe is empty, put NA in all columns
            if df.empty:
                df = pd.DataFrame(columns=['label',
                                           'area',
                                           'centroid-0',
                                           'centroid-1',
                                           'mean_intensity',
                                           'max_intensity',
                                           'min_intensity',
                                           'feret_diameter_max',
                                           'eccentricity',
                                           'orientation',
                                           'perimeter'])
                df.loc[0] = np.nan

            if self.addChannelComboBox.currentText() != "None":
                addprops = measure.regionprops_table(self.viewer.layers[i].data,
                                                     self.viewer.layers[self.addChannelComboBox.currentText()].data,
                                                     properties=['mean_intensity',
                                                                 'max_intensity',
                                                                 'min_intensity'])

                for j in addprops:
                    df[str(j) + '_AC'] = addprops[j]


            # Edit dataframe
            df["resolution"] = self.scaleBox.value()
            for i in self.TempDb:
                if i == "Area":
                    df["ROI_Area"] = self.TempDb[i]
                elif i == "Line":
                    df["Line_Length"] = self.TempDb[i]["length"][index]
                    df["Line_Width"] = self.TempDb[i]["width"][index]
                    df["Line_Area"] = self.TempDb[i]["area"][index]
                    df["bg_width"] = self.TempDb[i]["bg_width"][index]
                    df["bg_area"] = self.TempDb[i]["bg_area"][index]
                    df["bg_intensity"] = self.TempDb[i]["bg_intensity"][index]
                else:
                    df["ROI_Area"] = self.TempDb[i]["area"]
                    df["ROI_Length"] = self.TempDb[i]["length"]

            # Add to database and list
            conn = sqlite3.connect(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
            c = conn.cursor()

            # Check if table exists, if not create it
            tables = self.FetchFromSQL(c)
            print(tables)
            if tables == []:
                print("No tables exist, creating new table")
                df.to_sql(self.clusterComboBox.currentText() + "_0", conn, if_exists='fail', index=False)
                self.listList.addItem(self.clusterComboBox.currentText() + "_0")
            else:
                matches = [s for s in tables if self.clusterComboBox.currentText() in s]
                matches = [int(i.split("_")[1]) for i in matches]
                print(matches)
                n = findSmallestIntNotInList(matches)
                print(n)

                # Add to database
                df.to_sql(self.clusterComboBox.currentText() + "_" + str(n), conn, if_exists='fail', index=False)
                # Add to listWidget
                self.UpdateListWidget(self.FetchFromSQL(c))

            conn.commit()
            conn.close()

        self.TempDb = None

    # Save Results
    def saveResults(self):
        filelocation = QFileDialog.getSaveFileName(self, "Save Results", "")[0]
        if not os.path.isdir(filelocation):
            os.makedirs(filelocation)
        else:
            print("Folder already exists")
        conn = sqlite3.connect(f"{os.path.dirname(os.path.abspath(__file__))}/Temp.sqlite")
        c = conn.cursor()
        tables = self.FetchFromSQL(c)

        final_df = pd.DataFrame(columns=['Name',
                                         'Resolution (um/px)',
                                         'Number of Clusters',
                                         'Average Area (px)',
                                         'Average Area (um2)',
                                         'Average Diameter (px)',
                                         'Average Diameter (um)',
                                         'Mean Intensity',
                                         'Average Max Intensity',
                                         'Mean Intensity AC',
                                         'Average Max Intensity AC',
                                         'Average eccentricity',
                                         'ROI Area (px)',
                                         'ROI Area (um)',
                                         'ROI Area Length (px)',
                                         'ROI Area Length (um)',
                                         'ROI Line Length (px)',
                                         'ROI Line Length (um)',
                                         'ROI Line Width (px)',
                                         'ROI Line Width (um)',
                                         'ROI Line Area (px)',
                                         'ROI Line Area (um)',
                                         'ROI Line Background Width (px)',
                                         'ROI Line Background Width (um)',
                                         'ROI Line Background Area (px)',
                                         'ROI Line Background Area (um)',
                                         'ROI Line Background Intensity'])

        for i in tables:
            df = pd.read_sql(f"SELECT * FROM '{i}'", conn)
            #Save raw data
            df.to_csv(filelocation + "/" + i + ".csv", index=False)
            calc = {
                "Name": i,
                "Resolution (um/px)": df["resolution"].mean(),
                "Number of Clusters": (df["label"].max() if df["label"].max() > 0 else 0),
                "Average Area (px)": df["area"].mean(),
                "Average Area (um2)": df["area"].mean() * df["resolution"].mean() ** 2,
                "Average Diameter (px)": df["feret_diameter_max"].mean(),
                "Average Diameter (um)": df["feret_diameter_max"].mean() * df["resolution"].mean(),
                "Mean Intensity": df["mean_intensity"].mean(),
                "Average Max Intensity": df["max_intensity"].mean(),
                "Mean Intensity AC": (df["mean_intensity_AC"].mean() if "mean_intensity_AC" in df else np.nan),
                "Average Max Intensity AC": (df["max_intensity_AC"].mean() if "max_intensity_AC" in df else np.nan),
                "Average eccentricity": df["eccentricity"].mean(),
                "ROI Area (px)": (df["ROI_Area"].mean() if "ROI_Area" in df else np.nan),
                "ROI Area (um)": (df["ROI_Area"].mean() * df["resolution"].mean() ** 2 if "ROI_Area" in df else np.nan),
                "ROI Area Length (px)": (df["ROI_Length"].mean() if "ROI_Length" in df else np.nan),
                "ROI Area Length (um)": (df["ROI_Length"].mean() * df["resolution"].mean() if "ROI_Length" in df else np.nan),
                "ROI Line Length (px)": (df["Line_Length"].mean() if "Line_Length" in df else np.nan),
                "ROI Line Length (um)": (df["Line_Length"].mean() * df["resolution"].mean() if "Line_Length" in df else np.nan),
                "ROI Line Width (px)": (df["Line_Width"].mean() if "Line_Width" in df else np.nan),
                "ROI Line Width (um)": (df["Line_Width"].mean() * df["resolution"].mean() if "Line_Width" in df else np.nan),
                "ROI Line Area (px)": (df["Line_Area"].mean() if "Line_Area" in df else np.nan),
                "ROI Line Area (um)": (df["Line_Area"].mean() * df["resolution"].mean() ** 2 if "Line_Area" in df else np.nan),
                "ROI Line Background Width (px)": (df["bg_width"].mean() if "bg_width" in df else np.nan),
                "ROI Line Background Width (um)": (df["bg_width"].mean() * df["resolution"].mean() if "bg_width" in df else np.nan),
                "ROI Line Background Area (px)": (df["bg_area"].mean() if "bg_area" in df else np.nan),
                "ROI Line Background Area (um)": (df["bg_area"].mean() * df["resolution"].mean() ** 2 if "bg_area" in df else np.nan),
                "ROI Line Background Intensity": (df["bg_intensity"].mean() if "bg_intensity" in df else np.nan)



            }

            final_df = pd.concat([final_df, pd.DataFrame(calc, index=[0])])

        #Save to csv at filedialog location
        final_df.to_csv(filelocation + '/' + filelocation.split('/')[-1] + '.csv', index=False)






@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return ClusterCounter
