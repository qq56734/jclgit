x0 = 312
y0 = 171
x1 = 314
y1 = 171
x2 = 329
y2 = 181
x3 = 329
y3 = 177












dx1 = x1 - x0
dy1 = y1 - y0

dx2 = x3 - x2
dy2 = y3 - y2


D1 = x1 * y0 - x0 * y1
D2 = x3 * y2 - x2 * y3



y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
x = float(y * dx1 - D1) / dy1