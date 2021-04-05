import numpy as np


def _debug_print_bands(arr_print, chans_print= None):
    """Debug, print statistics for each band."""
    
    # Print all channels if no list is provided
    if chans_print is None:
        chans_print = range(0, arr_print.shape[2])
    print("Printing bands...")
    # Print
    for i_chan in chans_print:
        print(np.min(arr_print[:,:,i_chan]), 
        np.mean(arr_print[:,:,i_chan]), np.max(arr_print[:,:,i_chan]))


def _shift_im(t1, t2, changemap, load_options):
    """Shift images related to each other in row and colum.

        Only handles total shifts of even number of pixels.   
    """
    
    if load_options["debug"]:
        print("Shifting images...")
    
    if load_options["col_shift"] != 0:
        col_half = int(np.ceil(np.abs(load_options["col_shift"])/2) )
        col_v = int(2*col_half)
        changemap = changemap[:, col_half:-col_half, :]
        if load_options["col_shift"] > 0:
            if load_options["debug"]:
                print("COLUMN: Cropping ", col_v , " pixels from END of t1 and BEGINING of t2.")
            t1 = np.array(t1[:, :-col_v, :]) 
            t2 = np.array(t2[:, col_v:, :])      
        else:
            if load_options["debug"]:
                print("COLUMN: Cropping ", col_v , " pixels from BEGINING of t1 and END of t2.")
            t1 = np.array(t1[:, col_v:, :]) 
            t2 = np.array(t2[:, :-col_v, :])

    if load_options["row_shift"] != 0:
        row_half = int(np.ceil(np.abs(load_options["row_shift"])/2))
        row_v = int(2*row_half)
        changemap = changemap[row_half:-row_half, :, :]
        if load_options["row_shift"] > 0:
            if load_options["debug"]:
                print("ROW: Cropping ", row_v , " pixels from END of t1 and BEGINING of t2.")
            t1 = np.array(t1[:-row_v, :, :]) 
            t2 = np.array(t2[row_v:, :,  :])
        else:      
            if load_options["debug"]:
                print("ROW: Cropping ", row_v , " pixels from BEGINING of t1 and END of t2.")
            t1 = np.array(t1[row_v:, :, :]) 
            t2 = np.array(t2[:-row_v:, :,  :])

    return t1, t2, changemap


def _shift_2D(changemap, load_options):
    """Shift a 2D image in row and colum.

        Only handles total shifts of even number of pixels.   
    """
    
    if load_options["debug"]:
        print("Shifting 2D array...")
    
    if load_options["col_shift"] != 0:
        col_half = int(np.ceil(np.abs(load_options["col_shift"])/2) )
        changemap = changemap[:, col_half:-col_half]

    if load_options["row_shift"] != 0:
        row_half = int(np.ceil(np.abs(load_options["row_shift"])/2))
        changemap = changemap[row_half:-row_half, :]

    return changemap


def _norm01(input_array, norm_type="global", min_cap=None, max_cap=None, min_cap_value=np.NaN, max_cap_value=np.NaN):
    """Normalise data.
    
    Parameters:
    norm_type:
        'band' - min and max of each variable (column) is 0 and 1
        'global' - min of array is 0, max is 1
    min_cap: Truncate values below this value to min_cap_value before normalizing
    max_cap: Truncate values above this value to max_cap_value before normalizing
    """
    
    # Ensure that original input is not modified
    output_array = np.array(input_array, copy=True)
    
    # Replace values outside envolope/cap with NaNs (or specifie value)
    if min_cap is not None:
        output_array[output_array   < min_cap] = min_cap_value
                   
    if max_cap is not None:
        output_array[output_array  > max_cap] = max_cap_value
    
    # Normalise data for selected normalization option
    if norm_type.lower() in ["global", "all", "set"]:
        # Normalise to 0-1 (globally)
        output_array = input_array - np.nanmin(input_array)
        output_array = output_array/np.nanmax(output_array)
    elif norm_type.lower() in ["band", "channel"]:
        # Normalise to 0-1 for each channel (assumed to be last index)
        # Get shape of input
        input_shape = input_array.shape
        output_array = np.zeros(input_shape)
        # Normalise each channel
        for i_channel in range(0, input_shape[2]):
            output_array[:,:,i_channel] = input_array[:,:,i_channel] - np.nanmin(input_array[:,:,i_channel])
            output_array[:,:,i_channel] = output_array[:,:,i_channel]/np.nanmax(output_array[:,:,i_channel])
        
    return output_array


def _default_load_options(input_options=None):
    """
        Set default load options. 
        
        TODO: Different standard settings with input_options argument?
    """
    print("Using default load options!")
    load_options = dict()
    load_options["norm_type"] = "_clip"
    load_options["debug"] = False
    load_options["row_shift"] = int(0)
    load_options["col_shift"] = int(0)
    load_options["reduce"] = False

    return load_options


def _clip_numpy(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Reimplemented in numpy from _clip.

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """
    temp = np.copy(image)
    temp = np.reshape(temp, (-1, image.shape[-1]))

    limits = np.mean(temp, axis=0) + 3.0 * np.std(temp, 0)
    for i_chan, limit in enumerate(limits):
        channel = temp[:, i_chan]
        channel = np.clip(channel, 0, limit)
        ma, mi = np.max(channel), np.min(channel)
        channel = 2.0 * ((channel) / (ma)) - 1
        temp[:, i_chan] = channel

    return np.reshape(temp, image.shape)