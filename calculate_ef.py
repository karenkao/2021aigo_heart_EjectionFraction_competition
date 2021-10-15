import cv2
import math
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imutils import contours
from imutils import perspective
from scipy import signal
from scipy.spatial import distance as dist
from sklearn.metrics import mean_squared_error


def v(args):
    area, long_axis = args
    return (8*(area)**2)/(3*np.pi*long_axis)

def ef(ED, ES):
    EDV = v(ED)
    ESV = v(ES)
    SV = EDV-ESV
    return round(SV/EDV, 3)

# way1 最小外接框
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def mask_to_diameter_minAreaRect(mask):
    dim_long, dim_short, coord_long, coord_short = 0, 0, None, None

    # Preprocess mask
    mask = np.array(mask * 255, dtype=np.uint8)

    # find contours in the edge map
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    # loop over the contours individually
    for c in cnts:
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order
        box_order = perspective.order_points(box)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box_order
        
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dimA = round(dist.euclidean((tltrX, tltrY), (blbrX, blbrY)))
        dimB = round(dist.euclidean((tlblX, tlblY), (trbrX, trbrY)))

        # According to the guideline, we round the diameter, unit: mm
        coord_A = ((int(round(tltrX)), int(round(tltrY))), (int(round(blbrX)), int(round(blbrY))))
        coord_B = ((int(round(tlblX)), int(round(tlblY))), (int(round(trbrX)), int(round(trbrY))))

        # Take the largest one in the mask if nodule num > 1
        
        # Set long-axis and short-axis
        if dimA >= dimB:
            dim_long, coord_long = dimA, coord_A
            dim_short, coord_short = dimB, coord_B
        else:
            dim_long, coord_long = dimB, coord_B
            dim_short, coord_short = dimA, coord_A

    return dim_long, coord_long, dim_short, coord_short


# way2:找極值點，極左極右點、極上極下點
def mask_to_diameter_extremepoint(mask):
    mask = np.array(mask * 255, dtype=np.uint8)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    dim_long = round(dist.euclidean((extLeft[0], extLeft[1]), (extRight[0], extRight[1])))
    coord_long = (extLeft, extRight)
    dim_short = round(dist.euclidean((extTop[0], extTop[1]), (extBot[0], extBot[1])))
    coord_short= (extTop, extBot)
    return dim_long, coord_long, dim_short, coord_short

# way 3: 橢圓形法
def mask_to_diameter_ellipse(mask):
    mask = np.array(mask * 255, dtype=np.uint8)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(c)
    (xc, yc), (d1, d2), angle = ellipse

    # draw longaxis
    rmajor = max(d1, d2)/2
    if angle>90:
        angle = angle-90
    else:
        angle = angle+90
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    cv2.line(mask,(int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 0), 5)
    coord_long = ((int(xtop),int(ytop)), (int(xbot),int(ybot)))
    dim_long = round(dist.euclidean((int(xtop),int(ytop)), (int(xbot),int(ybot))))
    return dim_long, coord_long, None, None, ellipse
    
def plot_diameter(img, dim_long, dim_short, coord_long, coord_short, ellipse=None):
    mask = img.copy()
    #mask = np.array(mask * 255, dtype=np.uint8)
    
    cv2.line(mask, coord_long[0], coord_long[1], (0, 255, 255) ,5)
    cv2.putText(mask, str(dim_long), (coord_long[0][0]-3, coord_long[0][1]-3), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    if dim_short and coord_short:
        cv2.line(mask, coord_short[0], coord_short[1], (0, 255, 255) ,5)
        cv2.putText(mask, str(dim_short), (coord_short[0][0]-3, coord_short[0][1]-3), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv2.LINE_AA)
    if ellipse:
        cv2.ellipse(mask, ellipse, (255, 255,255), 3)
   
    return mask

def find_peak_valley(slice_area_longaxis_list):
    # user original area to find peak and valley
    data_y = [i[1] for i in slice_area_longaxis_list]
    global_peak = data_y.index(max(data_y))
    global_valley = data_y.index(min(data_y))

    data_y= np.array(data_y)
    data_x = np.array([i for i in range(len(data_y))])

    peak_indexes = signal.argrelextrema(data_y, np.greater, order=3)
    peak_indexes = peak_indexes[0]
    if not global_peak in peak_indexes:
        peak_indexes = list(peak_indexes)
        peak_indexes.append(global_peak)
        peak_indexes = np.array(peak_indexes)
    peak_indexes = sorted(peak_indexes)
    peak_x = peak_indexes
    peak_y = data_y[peak_indexes]


    valley_indexes = signal.argrelextrema(data_y, np.less, order=3)
    valley_indexes = valley_indexes[0]
    if not global_valley in valley_indexes:
        valley_indexes = list(valley_indexes)
        valley_indexes.append(global_valley)
        valley_indexes = np.array(valley_indexes)
    valley_indexes = sorted(valley_indexes)
    valley_x = valley_indexes
    valley_y = data_y[valley_indexes]
    return peak_x, valley_x, global_peak, global_valley

def plot_peak_valley(slice_area_longaxis_list):
    # use approximate area to plot peak valley figure
    data_y = [round(i[1], 3) for i in slice_area_longaxis_list]
    data_y= np.array(data_y)
    data_x = np.array([i for i in range(len(data_y))])
    peak_x, valley_x, global_peak_x, global_valley_x = find_peak_valley(slice_area_longaxis_list)
    peak_y = data_y[peak_x]
    valley_y = data_y[valley_x]
    (fig, ax) = plt.subplots()
    ax.plot(data_x, data_y)
    ax.scatter(peak_x, peak_y, marker='o', color='red', label='Peaks')
    ax.scatter(valley_x, valley_y, marker='o', color='green', label='valleys')
    plt.show()
    
def peak_valley_combinations(peak_x, valley_x, slice_area_longaxis_list):
    peak_valley_dict = {}
    for x in peak_x:
        peak_valley_dict[x] = {"type":"peak", "slice":slice_area_longaxis_list[x][0], 
                                       "area": slice_area_longaxis_list[x][1], 
                                       "long_axis": slice_area_longaxis_list[x][2]}
    for x in valley_x:
        peak_valley_dict[x] = {"type":"valley", "slice":slice_area_longaxis_list[x][0], 
                                       "area": slice_area_longaxis_list[x][1], 
                                       "long_axis": slice_area_longaxis_list[x][2]}
    sort_peak_valley = sorted(peak_x+valley_x)
    peak_valley_combination = []
    for i, index in enumerate(sort_peak_valley):
        start_index = index
        if i == len(sort_peak_valley)-1:
            continue
        end_index = sort_peak_valley[i+1]

        type_start = peak_valley_dict[start_index]['type']
        type_end = peak_valley_dict[end_index]['type']
        if type_start == type_end:
            continue

        index_a, slice_a, area_a, long_axis_a = start_index, peak_valley_dict[start_index]["slice"], peak_valley_dict[start_index]["area"], peak_valley_dict[start_index]["long_axis"]
        index_b, slice_b, area_b, long_axis_b = end_index, peak_valley_dict[end_index]["slice"], peak_valley_dict[end_index]["area"], peak_valley_dict[end_index]["long_axis"]
        if type_start == "peak":
            peak_valley_combination.append([[index_a, slice_a, area_a, long_axis_a],
                                                            [index_b, slice_b, area_b, long_axis_b ]])
        else:
            peak_valley_combination.append([[index_b, slice_b, area_b, long_axis_b],
                                                            [index_a, slice_a, area_a, long_axis_a]])
    return peak_valley_combination

def dynamically_avg_ef(ef_list):
    ef_list = sorted(ef_list, reverse=True)
    adj_ef_list =[None for i in range(len(ef_list))]
    for i in range(len(ef_list)):
        if i == len(ef_list)-1:
            continue
        mse = mean_squared_error([ef_list[i]], [ef_list[i+1]])
        
        if mse <= 0.005:
            adj_ef_list[i], adj_ef_list[i+1] = ef_list[i], ef_list[i+1]
    adj_ef_list = list(filter(lambda x: x != None, adj_ef_list))
    
    return sorted(adj_ef_list, reverse=True)

def automatically_calculate_ef(image_arr, 
                               mask_arr, 
                               if_show_longaxis = False, 
                               if_plot_peak_valley = True, 
                               show_gED_gES=True):
    
    # calculate slice, area, and longaxis via three different methods
    minAreaRec_slice_area_longaxis_list = []
    extremepoint_slice_area_longaxis_list = []
    ellipse_slice_area_longaxis_list = []
    for index in range(image_arr.shape[0]):
        img = image_arr[index]
        mask = mask_arr[index]
        area = mask.sum()
        if area == 0:
            continue
        dim_long, coord_long, dim_short, coord_short = mask_to_diameter_minAreaRect(mask)
        img_minAreaRect = plot_diameter(img, dim_long, dim_short, coord_long, coord_short)
        minAreaRec_slice_area_longaxis_list.append([index, area, dim_long])


        dim_long, coord_long, dim_short, coord_short = mask_to_diameter_extremepoint(mask)
        img_extremepoint = plot_diameter(img, dim_long, dim_short, coord_long, coord_short)
        extremepoint_slice_area_longaxis_list.append([index, area, dim_long])

        dim_long, coord_long, dim_short, coord_short , ellipse= mask_to_diameter_ellipse(mask)
        img_ellipse = plot_diameter(img, dim_long, dim_short, coord_long, coord_short, ellipse)
        ellipse_slice_area_longaxis_list.append([index, area, dim_long])
        if if_show_longaxis:
            plt.figure(figsize = (30, 40))
            plt.imshow(np.hstack((img_minAreaRect, img_extremepoint, img_ellipse)), cmap='gray')
            plt.imshow(np.hstack((mask,mask,mask)), cmap='cool', alpha=0.3)
            plt.show()
            
    # calculate ef based on three methods
    way_list = ["minAreaRec", "extremepoint", "ellipse"]
    slice_area_longaxis_list_all = [minAreaRec_slice_area_longaxis_list, 
                                    extremepoint_slice_area_longaxis_list, 
                                    ellipse_slice_area_longaxis_list]
    
    if if_plot_peak_valley:
        plot_peak_valley(slice_area_longaxis_list_all[0])
    
    result = {}
    for i, slice_area_longaxis_list in enumerate(slice_area_longaxis_list_all):
        way = way_list[i]
        ef_list = []
        peak_x, valley_x, global_peak_x, global_valley_x = find_peak_valley(slice_area_longaxis_list)
        peak_valley_combination = peak_valley_combinations(peak_x, valley_x, slice_area_longaxis_list)
        for beats in peak_valley_combination:
            ED = (beats[0][2], beats[0][3])
            ES = (beats[1][2], beats[1][3])
            EF = ef(ED, ES)
            start_end = sorted((beats[0][0], beats[1][0]))
            slice_ED, slice_ES = beats[0][1], beats[1][1]
            ef_list.append(EF)
        # global max, min 
        g_ED = (slice_area_longaxis_list[global_peak_x][1], slice_area_longaxis_list[global_peak_x][2])
        g_ES = (slice_area_longaxis_list[global_valley_x][1], slice_area_longaxis_list[global_valley_x][2])
        g_EF = ef(g_ED, g_ES)
        g_start_end = sorted((global_peak_x, global_valley_x))
        slice_gED, slice_gES =slice_area_longaxis_list[global_peak_x][0],slice_area_longaxis_list[global_valley_x][0]
        if not g_EF in ef_list:
            ef_list.append(g_EF)
        adj_ef_list = dynamically_avg_ef(ef_list)
        if len(adj_ef_list) >2:
            avg_ef = round(np.mean(dynamically_avg_ef(adj_ef_list)[:2]), 3)
        else:
            avg_ef = adj_ef_list[0]
        result[way] = {"slice_gED": slice_gED, "slice_gES": slice_gES, "avg_ef": avg_ef}
        if show_gED_gES:
            print(way)
            print("avg_ef:", avg_ef)
            plt.imshow(np.hstack((image_arr[slice_ED], image_arr[slice_ES])), cmap='gray')
            plt.imshow(np.hstack((mask_arr[slice_ED],mask_arr[slice_ES])), cmap='cool', alpha=0.3)
            plt.show()
            
    return result