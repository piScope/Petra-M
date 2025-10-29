def normalize(PETRAM_ARRAY_ID, PETRAM_ARRAY_COUNT, offset=0, norm=1.0):
    '''
    if array_id starts from 1 and ends with array_count

    array_id is normalized to the value from -norm+offset norm+offset 
    '''
    array_id = PETRAM_ARRAY_ID
    array_count = PETRAM_ARRAY_COUNT
    if array_count == 1:
        return offset

    n = (array_id-(array_count+1)/2)/((array_count-1)/2)
    return n*norm+offset
