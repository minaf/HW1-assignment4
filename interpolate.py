from math import floor
def interpolate(J, y, x):


	fl0 = int(floor(x))
	fl1 = int(floor(y))
	temp = (fl0+1-x)*J[fl1, fl0]- (fl0-x)*J[fl1, fl0+1]
	temp1 = (fl0+1-x)*J[fl1+1,fl0] - (fl0-x)*J[fl1+1, fl0+1]
	z1 = (fl1+1-y)*temp - (fl1-y)*temp1
	return z1
