#Need to install shapely first

from shapely.geometry import MultiPoint,Point,LineString
from shapely.geometry import shape
import numpy as np


def equation_line(x,y,entry):
  dx=x[1]-x[0]
  dy=y[1]-y[0]
  m=dy/dx
  return m*entry +y[0]- m*x[0]

def only_one(my_array):
  if len(my_array)>2:
    temp=np.zeros((2,2))
    temp[0]=my_array[0]
    temp[1]=my_array[len(my_array)-1]
    my_array=temp
  else:
    my_array=my_array
  return my_array   

for n in range(len(images)):
  image=images[n]
  centroid_x=int(np.mean(curly[n],axis=0)[0])
  centroid_y=int(np.mean(curly[n],axis=0)[1])
  start_point=(centroid_x, centroid_y)
  width=image.shape[0]
  height=image.shape[1]

  temp=[]
  for i in range(len(curly[n])):
    curly[n][i][0]=int(curly[n][i][0])
    curly[n][i][1]=int(curly[n][i][1])
    temp.append(curly[n][i])

  contour_points = MultiPoint(temp) 


  theta_5 = 30
  theta_5 *= np.pi/180.0
  end_point_5= (int(centroid_x+np.cos(theta_5)*height),int((centroid_y-np.sin(theta_5)*width)))


  theta_7 = 30
  theta_7 *= np.pi/180.0
  end_point_7= (int(centroid_x-np.cos(theta_7)*height),int((centroid_y-np.sin(theta_7)*width)))

  #compute the vertical line
  y_vert= np.linspace(0,512,512, dtype ='int')
  x_vert=np.full(y_vert.shape,fill_value=centroid_x,dtype=np.int)
  line1=[]
  for i in range(len(y_vert)):
    line1.append((x_vert[i],y_vert[i]))

  line_vert = MultiPoint(line1)

  #compute the horizontal line
  x_hor= np.linspace(0,512,512, dtype ='int')
  y_hor=np.full(x_hor.shape,fill_value=centroid_y,dtype=np.int)
  line2=[]
  for i in range(len(y_hor)):
    line2.append((x_hor[i],y_hor[i]))

  line_hor = MultiPoint(line2)

  #compute the first diagonal
  equation=np.zeros(512)
  temp=equation_line((centroid_x, end_point_5[0]), (centroid_y, end_point_5[1]),np.linspace(0,512,512, dtype ='int'))
  for i in range(len(temp)):
    equation[i]=int(temp[i])
  line3=[]
  x=np.linspace(0,512,512, dtype ='int')
  for i in range(len(x)):
    line3.append((x[i],equation[i]))

  line_diag1 = MultiPoint(line3)


  #compute the second diagonal
  equation2=np.zeros(512)
  temp2=equation_line((centroid_x, end_point_7[0]), (centroid_y, end_point_7[1]),np.linspace(0,512,512, dtype ='int'))
  for i in range(len(temp2)):
    equation2[i]=int(temp2[i])
  line4=[]
  x=np.linspace(0,512,512, dtype ='int')
  for i in range(len(x)):
    line4.append((x[i],equation2[i]))

  line_diag2 = MultiPoint(line4)

  #get the intersections points of the lines and the contour points
  inter_vert=line_vert.intersection(contour_points)
  inter_hor=line_hor.intersection(contour_points)
  inter_diag1=line_diag1.intersection(contour_points)
  inter_diag2=line_diag2.intersection(contour_points)


  #tranforming it into np array
  listarray1 = []
  for pp in inter_vert:
      listarray1.append([pp.x, pp.y])
  intersection_vertical = np.array(listarray1)

  #tranforming it into np array
  listarray2 = []
  for pp in inter_hor:
      listarray2.append([pp.x, pp.y])
  intersection_horizontal = np.array(listarray2)

  #tranforming it into np array
  listarray3 = []
  for pp in inter_diag1:
      listarray3.append([pp.x, pp.y])
  intersection_diag1 = np.array(listarray3)

  #tranforming it into np array
  listarray4 = []
  for pp in inter_diag2:
      listarray4.append([pp.x, pp.y])
  intersection_diag2 = np.array(listarray4)

  #take only the 2 intersections
  intersection_vertical=only_one(intersection_vertical)
  intersection_horizontal=only_one(intersection_horizontal)
  intersection_diag1=only_one(intersection_diag1)
  intersection_diag2=only_one(intersection_diag2)


  #store the control points on a list ,and save it in txt files
  liste=[]
  liste.append((intersection_vertical[0][0],intersection_vertical[0][1]))
  liste.append((intersection_vertical[1][0],intersection_vertical[1][1]))
  liste.append((intersection_horizontal[0][0],intersection_horizontal[0][1]))
  liste.append((intersection_horizontal[1][0],intersection_horizontal[1][1]))
  liste.append((intersection_diag1[0][0],intersection_diag1[0][1]))
  liste.append((intersection_diag1[1][0],intersection_diag1[1][1]))
  liste.append((intersection_diag2[0][0],intersection_diag2[0][1]))
  liste.append((intersection_diag2[1][0],intersection_diag2[1][1]))

  np.savetxt(PATH+str(n)+'.txt',liste)
