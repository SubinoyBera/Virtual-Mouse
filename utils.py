import numpy as np
import pyautogui

screen_width, screen_height= pyautogui.size()

def get_angle(a, b, c):
    """
    Calculate the angle between three points.
    Parameters:
    a : tuple
        The coordinates of the first point.
    b : tuple
        The coordinates of the second point.
    c : tuple
        The coordinates of the third point.

    Returns
    -------
    angle : float
        The angle in degrees between the three points.
    """
    radian= np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle= np.abs(np.degrees(radian))
    
    return angle

def get_distance(landmark_list) -> float:
    """
    The function computes the Euclidean distance between the first 
    two points in the `landmark_list` and scales it to a range of 0 to 1000.

    Parameters:
    landmark_list (list of tuples): A list containing at least two tuples, 
                                    each representing the (x, y) coordinates of a landmark.

    """

    if len(landmark_list)< 2:
        return
    
    (x1, y1), (x2, y2)= landmark_list[0], landmark_list[1]
    d= np.hypot(x2-x1, y2-y1)

    return np.interp(d, [0,1], [0,1000])

def move_mouse(index_fingertip):    
    """
    Moves the mouse pointer to the screen coordinates corresponding to the 
    position of the index fingertip.
    """
    if index_fingertip is not None:
        x= int(index_fingertip.x * screen_width)
        y= int(index_fingertip.y * screen_height)
        pyautogui.moveTo(x,y)