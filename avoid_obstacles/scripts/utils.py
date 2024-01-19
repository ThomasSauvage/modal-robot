def show_dists(ranges: 'list[float]'):
    for i in range(10):
        dist = ranges[i*len(ranges)//10]
        if dist < float('+inf'):
            print(str(round(100*dist)).ljust(4), end='')
        else:
            print('inf ',end='')
    
    print()


def get_dist_dir(data, angle: float) -> float:
    ''' Return the distance to the closest object
        in the direction angle.

        - Angle unit: Rad

        - 0 is in front
        - pi is behind
    '''

    if not data.angle_min <= angle <= data.angle_max:
        raise ValueError('Angle asked out of LIDAR range')
    

    i =  int((angle - data.angle_min) / data.angle_increment)

    return data.ranges[i]