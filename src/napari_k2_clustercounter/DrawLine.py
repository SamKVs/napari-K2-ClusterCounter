import numpy as np
from skimage.draw import polygon, disk

def draw_path(path, width, canvas):
    print(canvas.shape)
    stack = np.stack((path[0:-1], path[1:]), axis=1)
    print(stack)
    rr = np.array([])
    cc = np.array([])
    for line in stack:
        def transformer(P0, Pref, width):
            Pref = np.subtract(Pref, P0)
            Pnew = (Pref / np.sqrt(Pref.dot(Pref))) * (width / 2)
            Pturn = np.array([np.matmul(Pnew, np.array([[0, 1], [-1, 0]])),
                               np.matmul(Pnew, np.array([[0, -1], [1, 0]]))])
            Pturned = np.add(Pturn, P0)
            return Pturned

        p1 = line[0]
        p2 = line[1]
        p1turned = transformer(p1,p2,width)
        p2turned = transformer(p2,p1,width)
        polygoncoords = np.concatenate((p1turned,p2turned))
        r, c = polygon(polygoncoords[:,1],polygoncoords[:,0])
        rr = np.concatenate((rr,r))
        cc = np.concatenate((cc,c))

    for point in path[1:-1]:
        c, r = disk(point, width/2)
        rr = np.concatenate((rr,r))
        cc = np.concatenate((cc,c))

    coords = np.stack((rr,cc),axis=1).astype(int)

    #Clean coords that fall outside the canvas
    coords = coords[(coords[:,0] >= 0) & (coords[:,0] < canvas.shape[0]) & (coords[:,1] >= 0) & (coords[:,1] < canvas.shape[1])]

    canvas[coords[:,1],coords[:,0]] = 1

    return canvas











