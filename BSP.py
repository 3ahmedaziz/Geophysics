import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy import signal
import matplotlib.dates as mdates
from dateutil.parser import parse
from obspy.io.segy.segy import _read_segy
from sklearn import preprocessing
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy import interpolate

class BSP():
    '''
    A class to organize, process, and visualize segy and HDF5 seismic data 
    
    Usage: Data = BSP.BSP() (after importing BSP) 
    
    Input: 
        None
        
    Attributes:
        version: Class release version. Type: int
        start_time: Time of DAS recording. Type: Datetime
        freq: Sampling frequency in the DAS data. Type: float
        data: (M,N) Array holding DAS data, M: Number of samples, N: Number of traces. Type: numpyArray
        channel: (N,1) Array hodling the channel numbers. Type: numpyArray
        time: (M, 1) Array holding the time axis for the DAS data. 
        
    Methods: 
        I/O:
        load_segy: Read segy file and set class attributes
        load_hdf5: Read hdf5 file and set class attributes
        save_hdf5: Save class data to hdf5 file
        copy: Create a deep copy of class 
        
        Filters: 
        lp_fltr: Apply low-pass filter on data in either axes, t or x
        hp_fltr: Apply high-pass filter on data in either axes, t or x
        bp_fltr: Apply band-pass filter on data in either axes, t or x
        median_fltr: Apply median filter on data in either axes, t or x
        fk_fltr: Apply f-k filter on data based on two velocities
        fk_lp_fltr: Apply low-pass filter on data data in fk-domain (airy-disk) 
        fk_hp_fltr: Apply high-pass filter on data data in fk-domain (airy-disk) 
        fk_bp_fltr: Apply band-pass filter on data data in fk-domain (airy-disk) 
        gaussian_fltr: Apply Gaussian filter on data
        derivative_fltr: Apply first derivative filter on data 
        
        Operations:
        standarize: Standarize data with zero mean and unit standard deviation
        interpolate: Interpolate data set with new spacing
        convolve_data: Convolve entire data with a source in either axes, t or x
        convolve_trace: Return a single trace convolution with a source in either axes, t or x
        get_trace_fft: Return freuency range and fft of a single trace of either axes, t or x
        get_fft2d: Return 2D fft of data
        
        Visualization: 
        plt_trace: Plot (matplotlib) a single trace in either axes, t or x
        plt_trace_fft: Plot (matplotlib) fft of a single trace in either axes, t or x
        plt_fft2d: Plot (matplotlib) 2D fft of data 
        plt_trace_spectrogram: Plot (matplotlib) spectrogram of a single trace in either axes, t or x
        Plot (matplotlib) a simple waterfall plot of the data

        
    Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
    '''
    def __init__(self):
               
        self.version = '1.0' 
        self.date_time = datetime.strptime('2020 1 1 1 1', '%Y %j %H %M %S') # Done
        self.freq = 0.0 
        self.dx = 1.0 
        self.k    = 1/self.dx 
        self.data = np.array([]) 
        self.channel = np.array([]) 
        self.time = np.array([]) 
        self.depth = np.array([]) 
        self.coordinates_x = np.array([])  
        self.coordinates_y = np.array([]) 
        
        
    def load_segy(self, file_name): 
        '''
        Read segy file and set class attributes
        
        Usage: Data.load_segy('file_name.segy')
        
        Input: 
            file_name: File name. Type: string
            
        Output:
            None

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        stream = _read_segy(file_name, headonly=True) 
        
        self.freq = 1/stream.traces[0].header.sample_interval_in_ms_for_this_trace*1e6
        npts = stream.traces[0].npts 
        time = npts/self.freq
        
        self.time = np.arange(0, time, 1/self.freq)
        self.data = np.vstack([trace.data for trace in stream.traces]).T
        self.channel = np.arange(0, self.data.shape[1],1) 
        self.depth = self.channel*self.dx
        
        date_time = (str(stream.traces[0].header.year_data_recorded) + ' ' + str(stream.traces[0].header.day_of_year) +
             ' ' + str(stream.traces[0].header.hour_of_day) + ' ' + str(stream.traces[0].header.minute_of_hour) + 
             ' ' + str(stream.traces[0].header.second_of_minute))

        self.date_time = datetime.strptime(date_time, '%Y %j %H %M %S')
        
        self.coordinates_x = np.array([trace.header.group_coordinate_x for trace in stream.traces])
        self.coordinates_y = np.array([trace.header.group_coordinate_x for trace in stream.traces])

    def load_hdf5(self, file_name): 
        '''
        Read hdf5 file and set class attributes
        
        Usage: Data.load_hdf5('file_name.hdf5')
        
        Input: 
            file_name: File name. Type: string
            
        Output:
            None

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        file = h5py.File(file_name, 'r')
        
        self.date_time = datetime.strptime(np.array(file['Start Time']).flatten()[0], "%Y-%m-%d %H:%M:%S")
        self.freq = np.array(file['Frequency (Hz)']).flatten()[0]
        self.dx = np.array(file['Spacing (m)']).flatten()[0]
        self.k    = np.array(file['Wavenumber']).flatten()[0]
        self.data = np.array(file['Data'])
        self.channel = np.array(file['Channels'])
        self.time = np.array(file['Time (s)'])
        self.depth = np.array(file['Depth (m)'])
        self.coordinates_x = np.array(file['X Coordinate']) 
        self.coordinates_y = np.array(file['Y Coordinate']) 
                
        
    def save_hdf5(self, file_name): 
        '''
        Save class data to hdf5 file
        
        Usage: Data.save_hdf5('file_name')
        
        Input: 
            file_name: File name. Type: string
            
        Output:
            None

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        data_file = h5py.File(file_name+'.hdf5', 'w')
        data_file.create_dataset('Data', data=self.data)
        data_file.create_dataset('Time (s)', data=self.time) 
        data_file.create_dataset('Depth (m)', data=self.depth) 
        data_file.create_dataset('Channels', data=self.channel) 
        data_file.create_dataset('X Coordinate', data=self.coordinates_x) 
        data_file.create_dataset('Y Coordinate', data=self.coordinates_y) 
        
        dtype = h5py.special_dtype(vlen=str)
        data_file.create_dataset('Start Time', data=str(self.date_time), dtype=dtype) 
        data_file.create_dataset('Frequency (Hz)', data=self.freq) 
        data_file.create_dataset('Wavenumber', data=self.k) 
        data_file.create_dataset('Spacing (m)', data=self.dx) 

        data_file.close()
    
    def copy(self): 
        '''
        Create a deep copy of class 
        
        Usage: Data_copy = Data.copy()
        
        Input: 
            None
            
        Output:
            Deep copy of class

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        return deepcopy(self)
    
    
    def lp_fltr(self, freq, axis='t', order=2, in_place=False, fltr_response=False, worN=1024, h=5, w=12):
        '''
        Apply low-pass filter on data in either axes, t or x 
        
        Usage: * Applying filter on the same data: 
                    Data.lp_fltr(freq, in_place=True)
                * Applying filter on the same data (x axis): 
                    Data.bp_fltr(freq, axis='x', in_place=True)
               * Getting a deep copy with filtered data :
                    filtered_Data = Data.lp_fltr(freq)
        
        Input: 
            freq: Cut-off frequency
            axis: Filtering axis; 't' for time (default), 'x' for space
            order: Butterworth filter order (default = 2)
            in_place: Apply filter on the data itself (True) or return a deep copy with the filtered data (default = False)
            fltr_response: Plot the filter response of the filter (default = False) - Only when in_place=True
            worN: Frequency range for the filter response (default = 1024)
            h: Height of the filter response plot (default = 5)
            w: Width of the filter response plot (default = 12)
            
        Output:
            in_place = True and fltr_response = False: None 
            in_place = False: a class deep copy with the filtered data 
            in_place = True and fltr_response = True: Filter response 
            

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
            
            b, a = signal.butter(order, 2*freq/self.freq, 'low')

            if (in_place):
                for i in range(self.data.shape[1]):
                    self.data[:,i] = signal.filtfilt(b,a,self.data[:,i])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[1]):
                    filtered_data[:,i] = signal.filtfilt(b,a,self.data[:,i])
                
                copy = self.copy()
                copy.data = filtered_data
                    
                return copy
            
            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Frequency (Hz)')
                plt.grid(True)
                plt.show()


        elif (axis == 'x'): 
            b, a = signal.butter(order, 2*freq/self.k, 'low')

            if (in_place):
                for i in range(self.data.shape[0]):
                    self.data[i,:] = signal.filtfilt(b,a,self.data[i,:])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[0]):
                    filtered_data[i,:] = signal.filtfilt(b,a,self.data[i,:])

                copy = self.copy()
                copy.data = filtered_data
                    
                return copy

            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Frequency (Hz)')
                plt.grid(True)
                plt.show()
        
    def hp_fltr (self, freq, axis='t', order=2, in_place=False, fltr_response=False, worN=1024, h=5, w=12): 
        '''
        Apply high-pass filter on data in either axes, t or x 
        
        Usage: * Applying filter on the same data (time axis): 
                    Data.hp_fltr(freq, in_place=True)
                * Applying filter on the same data (x axis): 
                    Data.hp_fltr(freq, axis='x', in_place=True)
               * Getting a deep copy with filtered data :
                    filtered_Data = Data.hp_fltr(freq)
        
        Input: 
            freq: Cut-off frequency
            axis: Filtering axis; 't' for time (default), 'x' for space
            order: Butterworth filter order (default = 2)
            in_place: Apply filter on the data itself (True) or return a deep copy with the filtered data (default = False)
            fltr_response: Plot the filter response of the filter (default = False) - Only when in_place=True
            worN: Frequency range for the filter response (default = 1024)
            h: Height of the filter response plot (default = 5)
            w: Width of the filter response plot (default = 12)
            
        Output:
            in_place = True and fltr_response = False: None 
            in_place = False: a class deep copy with the filtered data 
            in_place = True and fltr_response = True: Filter response 
            

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
            
            b, a = signal.butter(order, 2*freq/self.freq, 'high')

            if (in_place):
                for i in range(self.data.shape[1]):
                    self.data[:,i] = signal.filtfilt(b,a,self.data[:,i])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[1]):
                    filtered_data[:,i] = signal.filtfilt(b,a,self.data[:,i])
                    
                copy = self.copy()
                copy.data = filtered_data
                    
                return copy
            
            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Frequency (Hz)')
                plt.grid(True)
                plt.show()


        elif (axis == 'x'): 
            b, a = signal.butter(order, 2*freq/self.k, 'high')

            if (in_place):
                for i in range(self.data.shape[0]):
                    self.data[i,:] = signal.filtfilt(b,a,self.data[i,:])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[0]):
                    filtered_data[i,:] = signal.filtfilt(b,a,self.data[i,:])
                    
                copy = self.copy()
                copy.data = filtered_data
                    
                return copy
            
            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Frequency (Hz)')
                plt.grid(True)
                plt.show()

            
    def bp_fltr (self, freq1, freq2, axis='t', order=2, in_place=False, fltr_response=False, worN=1024, h=5, w=12): 
        '''
        Apply band-pass filter on data in either axes, t or x 
        
        Usage: * Applying filter on the same data (time axis): 
                    Data.bp_fltr(freq1,freq2, in_place=True)
                * Applying filter on the same data (x axis): 
                    Data.bp_fltr(freq1,freq2, axis='x', in_place=True)
               * Getting a deep copy with filtered data :
                    filtered_Data = Data.bp_fltr(freq1,freq2)
        
        Input: 
            freq1: Lower cut-off frequency
            freq2: Upper cut-off frequency
            axis: Filtering axis; 't' for time (default), 'x' for space
            order: Butterworth filter order (default = 2)
            in_place: Apply filter on the data itself (True) or return a deep copy with the filtered data (default = False)
            fltr_response: Plot the filter response of the filter (default = False) - Only when in_place=True
            worN: Frequency range for the filter response (default = 1024)
            h: Height of the filter response plot (default = 5)
            w: Width of the filter response plot (default = 12)
            
        Output:
            in_place = True and fltr_response = False: None 
            in_place = False: a class deep copy with the filtered data 
            in_place = True and fltr_response = True: Filter response 
            

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
            
            b, a = signal.butter(order, (2*freq1/self.freq, 2*freq2/self.freq), 'bandpass')

            if (in_place):
                for i in range(self.data.shape[1]):
                    self.data[:,i] = signal.filtfilt(b,a,self.data[:,i])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[1]):
                    filtered_data[:,i] = signal.filtfilt(b,a,self.data[:,i])
                    
                copy = self.copy()
                copy.data = filtered_data
                    
                return copy
            
            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Frequency (Hz)')
                plt.grid(True)
                plt.show()


        elif (axis == 'x'): 
            b, a = signal.butter(order, (2*freq1/self.k, 2*freq2/self.k), 'bandpass')

            if (in_place):
                for i in range(self.data.shape[0]):
                    self.data[i,:] = signal.filtfilt(b,a,self.data[i,:])
            else: 
                filtered_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[0]):
                    filtered_data[i,:] = signal.filtfilt(b,a,self.data[i,:])
                    
                copy = self.copy()
                copy.data = filtered_data
                    
                return copy
            
            if (fltr_response): 
                W, H = signal.freqz(b,a,worN=worN)
                
                plt.figure(figsize=(w, h))
                plt.plot(W*self.freq/(2*np.pi), 20 * np.log10(np.abs(H)), 'b')
                plt.xscale('log')
                plt.ylabel('Amplitude (dB)', color='b')
                plt.xlabel('Spatial Frequency (cy/m)')
                plt.grid(True)
                plt.show()

            
    def get_trace_fft(self, trace_number, axis='t'):
        '''
        Return frequency range and fft of a single trace of either axes, t or x
        
        Usage: * Getting the fft (time axis) of a trace: 
                    freq , fft = Data.get_trace_fft(trace_number)
               * Getting the fft (x axis) of a trace: 
                    freq , fft = Data.get_trace_fft(trace_number, axis='t')
        
        Input: 
            trace_number: Trace number
            axis: Fft axis; 't' for time (default), 'x' for space 
            
        Output:
            frequency range and fft arrays of a specified trace 
            

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
            fft = np.fft.fftshift(np.fft.fft(self.data[:,trace_number]))/(self.data.shape[0])
            fft_freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,trace_number]),1/self.freq))
            
            return fft_freq, fft
        
        elif (axis == 'x'): 
            fft = np.fft.fftshift(np.fft.fft(self.data[trace_number, :]))/(self.data.shape[1])
            fft_freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[trace_number, :]),1/self.k))
        
            return fft_freq, fft
            
            
    def plt_trace(self, trace_number, axis='t', h=5, w=15):
        '''
        Plot (matplotlib) a single trace in either axes, t or x 
        
        Usage: * Time-axis: Data.plt_trace(trace_number)
               * x-axis: Data.plt_trace(trace_number, axis='x')
        
        Input: 
            trace_number: Trace number
            axis: plotting axis; 't' for time (default) and 'x' for spcace
            h: Plot height (default = 5) 
            w: Plot width (default = 15)
            
        Output:
            None 

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis =='t'): 
            
            plt.figure(figsize=(w, h))
            plt.plot(self.time,self.data[:,trace_number])
            plt.xlabel('Time (s)',fontsize=16)
            plt.ylabel('Amplitude',fontsize=16)
            plt.grid()
        
            plt.tight_layout()
            plt.show()
            
        elif (axis =='x'): 
            plt.figure(figsize=(w, h))
            plt.plot(self.depth,self.data[trace_number,:])
            plt.xlabel('Distance (m)',fontsize=16)
            plt.ylabel('Amplitude',fontsize=16)
            plt.grid()
        
            plt.tight_layout()
            plt.show()

    
    def plt_trace_fft(self, trace_number, axis='t', h=5, w=15): 
        '''
        Plot (matplotlib) fft of a single trace in either axes, t or x 
        
        Usage: * Time-axis: Data.plt_trace_fft(trace_number)
               * x-axis: Data.plt_trace_fft(trace_number, axis='x')
        
        Input: 
            trace_number: Trace number
            axis: plotting axis; 't' for time (default) and 'x' for spcace
            h: Plot height (default = 5) 
            w: Plot width (default = 15)
            
        Output:
            None 

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
        
            fft = np.fft.fftshift(np.abs(np.fft.fft(self.data[:,trace_number])))/(self.data.shape[0])
            fft_freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,trace_number]),d=1/self.freq))
        
            plt.figure(figsize=(w, h))
            plt.plot(fft_freq,fft)
            plt.xlabel('Frequency (Hz)',fontsize=16)
            plt.ylabel('Amplitude',fontsize=16)
            plt.grid()
    
            plt.tight_layout() 
            plt.show()

        elif (axis == 'x'):
        
            fft = np.fft.fftshift(np.abs(np.fft.fft(self.data[trace_number,:])))/(self.data.shape[1])
            fft_freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[trace_number,:]),d=1/self.k))
        
            plt.figure(figsize=(w, h))
            plt.plot(fft_freq,fft)
            plt.xlabel('Wavenumber (1/m)',fontsize=16)
            plt.ylabel('Amplitude',fontsize=16)
            plt.grid()
    
            plt.tight_layout() 
            plt.show()

        
    def plt_trace_spectrogram(self, trace_number, axis='t', nlen=256, noverlap=256/2, vmax=0.01,  h=5, w=15):
        '''
        Plot (matplotlib) spectrogram of a single trace in either axes, t or x
        
        Usage: * Time-axis: Data.plt_trace_spectrogram(trace_number)
               * x-axis: Data.plt_trace_spectrogram(trace_number, axis='x')
        
        Input: 
            trace_number: Trace number
            nlen: Spectrogram window size (default = 256)
            noverlap: Spectrogram window overlapping size (default = 246/2)
            vmax: vmax value for plt.pcolormesh (default = 0.01)
            h: Plot height (default = 5) 
            w: Plot width (default = 15)
            
        Output:
            None.  
            

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis =='t'): 
            f, t, S = signal.spectrogram(self.data[:,trace_number],self.freq,nperseg=nlen,noverlap=noverlap)

            plt.figure(figsize=(w,h)) 
            plt.pcolormesh(t,f,S,cmap='viridis', vmax=vmax)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Amplitude')

            plt.tight_layout()
            plt.show()
            
        elif (axis =='x'): 
            k, x, S = signal.spectrogram(self.data[trace_number,:],self.k,nperseg=nlen,noverlap=noverlap)

            plt.figure(figsize=(w,h)) 
            plt.pcolormesh(x,k,S,cmap='viridis', vmax=vmax)
            plt.xlabel('Distance (m)')
            plt.ylabel('Wavenumber (1/m)')
            plt.colorbar(label='Amplitude')

            plt.tight_layout()
            plt.show()
            
        
    def convolve_data(self, source, axis='t', mode='same', in_place=False):
        '''
        Convolve entire data with a source in either axes, t or x
        
        Usage: * Convolving the same data in time axis: 
                    Data.convolve_data(source, in_place=True)
                * Convolving the same data in x axis: 
                    Data.convolve_data(source, axis='x', in_place=True)
               * Getting a class deep copy with the convolved data:
                    filtered_Data = Data.convolve_data(source)
        
        Input: 
            source: The source the data should be convolved with
            mode: Convlution mode (default = 'same')
            in_place: Apply Convolution on the data itself (True) or return a class copy with the convolved data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: Class copy with the convolved data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis=='t'): 
            
            if (in_place):
                for i in range(self.data.shape[1]):
                    self.data[:,i] = np.convolve(self.data[:,i], source, mode=mode) 
            else: 
                convolved_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[1]):
                    convolved_data[:,i] = np.convolve(self.data[:,i], source, mode=mode)
                    
                copy = self.copy()
                copy.data = convolved_data
                    
                return copy
        
        elif (axis=='x'): 
            
            if (in_place):
                for i in range(self.data.shape[1]):
                    self.data[i,:] = np.convolve(self.data[i,:], source, mode=mode) 
            else: 
                convolved_data = np.zeros(self.data.shape)
                for i in range(self.data.shape[1]):
                    convolved_data[i,:] = np.convolve(self.data[i,:], source, mode=mode)
                    
                copy = self.copy()
                copy.data = convolved_data
                    
                return copy  
            
    def convolve_trace(self, source, trace_number, axis='t', mode='same'):
        '''
        Return a single trace convolution with a source in either axes, t or x
        
        Usage: convolved_trace = Data.convolve_trace(source, trace_number)
        
        Input: 
            trace_number: Trace number 
            source: The source the data should be convolved with
            mode: Convlution mode (default = 'same')
            
        Output:
            Convolved trace
            
        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (axis == 't'):
            return np.convolve(self.data[:,trace_number], source, mode=mode)
        
        elif (axis == 'x'): 
            return np.convolve(self.data[trace_number,:], source, mode=mode)            
    
    def median_fltr(self, kernel=None, in_place=False):
        '''
        Apply median filter on data in either axes, t or x
        
        Usage: * filtering the same data: 
                    Data.median_fltr(in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.median_fltr()
        
        Input: 
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            kernel: The median filter window 
            
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (in_place): 
            for i in range(self.data.shape[1]):
                self.data[:,i] = signal.medfilt(self.data[:,i], kernel_size=kernel) 
        
        else:
            filtered_data = np.zeros(self.data.shape)
            for i in range(self.data.shape[1]):
                filtered_data[:,i] = signal.medfilt(self.data[:,i], kernel_size=kernel) 
            
            copy = self.copy()
            copy.data = filtered_data

            return copy            

    def get_fft2d (self):
        '''
        Return 2D fft of data
        
        Usage: * fft2d = Data.get_fft2d()
        
        Input: 
            None
            
            
        Output:
           2D fft array of data
           
        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        return np.fft.fftshift(np.fft.fft2(self.data)/(self.data.shape[0]*self.data.shape[1]))

                    
    def plt_fft2d(self, cmap='jet', vmax=1, h=8, w=10, aspect='auto'): 
        '''
        Plot (matplotlib) 2D fft of the data set
        
        Usage: * Data.plt_fft2d()
        
        Input: 
            cmap: color style (default = 'jet')
            vmax: vmax value for plt.imshow (default = 1)
            h: Plot height (default = 5) 
            w: Plot width (default = 15)
            aspect: aspect ratio of the axes (default = 'auto')
            
        Output:
            None.  
            
        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        fft2d = np.fft.fftshift(np.fft.fft2(self.data)/(self.data.shape[0]*self.data.shape[1]))

        plt.figure(figsize=(w,h)) 
        plt.imshow(np.abs(fft2d),extent=[-self.k/2, self.k/2, -self.freq/2, self.freq/2], cmap=cmap, vmax=vmax, aspect=aspect)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Wavenumber (1/m)')
        plt.colorbar(label='Amplitude')

        plt.tight_layout()
        plt.show()
        
    def fk_fltr(self, v1, v2, in_place=False): 
        '''
        Apply f-k filter on data based on two velocities
        
        Usage: * filtering the same data: 
                    Data.fk_fltr(v1, v2, in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.fk_fltr(v1, v2)
        
        Input: 
            v1 : Lower cut-off velocity 
            v2 : Upper cut-off velocity 
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''                 
        nt, nx = self.data.shape[0], self.data.shape[1]
        fft2d = np.fft.fftshift(np.fft.fft2(self.data))
                                
        freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,0]),d=1/self.freq))
        k    = np.fft.fftshift(np.fft.fftfreq(len(self.data[0,:]),d=1/self.k))

        fft2d_magnitude = np.abs(fft2d)
        fft2d_phase     = np.angle(fft2d)

        pie = np.zeros((nt, nx))
        
        for tt in range(nt):
            for xx in range(nx):
                if (k[xx] == 0): pie[tt,xx] = 0; 
                elif ((freq[tt])/(-k[xx]) > v1 and (freq[tt])/(-k[xx]) < v2):
                    pie[tt,xx] = 1
                
        filtered_fft2d_magnitude = fft2d_magnitude*pie
        filtered_fft2d           = filtered_fft2d_magnitude*np.exp(1j*fft2d_phase)
        filtered_data            = np.real(np.fft.ifft2(np.fft.fftshift(filtered_fft2d)))

        if (in_place):
            self.data = filtered_data
                    
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy
        
    def fk_lp_fltr(self, ff, kk,  in_place=False):
        '''
        Apply low-pass filter on data in fk-domain (airy-disk) 
        
        Usage: * filtering the same data: 
                    Data.fk_lp_fltr(ff, kk, in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.fk_lp_fltr(ff, kk)
        
        Input: 
            ff : Frequency coordinate for the airy-disk radius (cut-off frequency) 
            kk : Wavenumber coordinate for the airy-disk radius (cut-off wavenumber)
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''                 
        nt, nx = self.data.shape[0], self.data.shape[1]
        fft2d = np.fft.fftshift(np.fft.fft2(self.data))
                                
        freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,0]),d=1/self.freq))
        k    = np.fft.fftshift(np.fft.fftfreq(len(self.data[0,:]),d=1/self.k))

        fft2d_magnitude = np.abs(fft2d)
        fft2d_phase     = np.angle(fft2d)
        
        radius = np.sqrt((kk)**2+(ff)**2)
        disk = np.zeros((nt, nx))
        
        for tt in range(nt):
            for xx in range(nx):
                if (np.sqrt(((k[xx])**2+(freq[tt])**2)) < radius):
                    disk[tt,xx] = 1
                
        filtered_fft2d_magnitude = fft2d_magnitude*disk
        filtered_fft2d           = filtered_fft2d_magnitude*np.exp(1j*fft2d_phase)
        filtered_data            = np.real(np.fft.ifft2(np.fft.fftshift(filtered_fft2d)))

        if (in_place):
            self.data = filtered_data
                                
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy
        
    def fk_hp_fltr(self, ff, kk, in_place=False):
        '''
        Apply high-pass filter on data in fk-domain (airy-disk) 
        
        Usage: * filtering the same data: 
                    Data.fk_hp_fltr(ff, kk, in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.fk_hp_fltr(ff, kk)
        
        Input: 
            ff : Frequency coordinate for the airy-disk radius (cut-off frequency) 
            kk : Wavenumber coordinate for the airy-disk radius (cut-off wavenumber)
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''        
        nt, nx = self.data.shape[0], self.data.shape[1]
        fft2d = np.fft.fftshift(np.fft.fft2(self.data))
                                
        freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,0]),d=1/self.freq))
        k    = np.fft.fftshift(np.fft.fftfreq(len(self.data[0,:]),d=1/self.k))

        fft2d_magnitude = np.abs(fft2d)
        fft2d_phase     = np.angle(fft2d)
        
        radius = np.sqrt((kk)**2+(ff)**2)
        disk = np.zeros((nt, nx))
        
        for tt in range(nt):
            for xx in range(nx):
                if (np.sqrt(((k[xx])**2+(freq[tt])**2)) > radius):
                    disk[tt,xx] = 1
                
        filtered_fft2d_magnitude = fft2d_magnitude*disk
        filtered_fft2d           = filtered_fft2d_magnitude*np.exp(1j*fft2d_phase)
        filtered_data            = np.real(np.fft.ifft2(np.fft.fftshift(filtered_fft2d)))

        if (in_place):
            self.data = filtered_data
                                
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy    
                                
    def fk_bp_fltr(self, ff1, kk1, ff2, kk2, in_place=False):
        '''
        Apply band-pass filter on data in fk-domain (airy-disk) 
        
        Usage: * filtering the same data: 
                    Data.fk_bp_fltr(ff1, kk1, ff2, kk2, in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.fk_bp_fltr(ff1, kk1, ff2, kk2)
        
        Input: 
            ff1 : Frequency coordinate for the first airy-disk radius (lower cut-off frequency) 
            kk1 : Wavenumber coordinate for the first airy-disk radius (lower cut-off wavenumber)
            ff1 : Frequency coordinate for the second airy-disk radius (upper cut-off frequency) 
            kk1 : Wavenumber coordinate for the second airy-disk radius (upper cut-off wavenumber)
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''        
        nt, nx = self.data.shape[0], self.data.shape[1]
        fft2d = np.fft.fftshift(np.fft.fft2(self.data))
                                
        freq = np.fft.fftshift(np.fft.fftfreq(len(self.data[:,0]),d=1/self.freq))
        k    = np.fft.fftshift(np.fft.fftfreq(len(self.data[0,:]),d=1/self.k))

        fft2d_magnitude = np.abs(fft2d)
        fft2d_phase     = np.angle(fft2d)
        
        radius1 = np.sqrt((kk1)**2+(ff1)**2)
        radius2 = np.sqrt((kk2)**2+(ff2)**2)

        disk = np.zeros((nt, nx))
        
        for tt in range(nt):
            for xx in range(nx):
                if (np.sqrt(((k[xx])**2+(freq[tt])**2)) > radius1 and np.sqrt(((k[xx])**2+(freq[tt])**2)) < radius2):
                    disk[tt,xx] = 1
                
        filtered_fft2d_magnitude = fft2d_magnitude*disk
        filtered_fft2d           = filtered_fft2d_magnitude*np.exp(1j*fft2d_phase)
        filtered_data            = np.real(np.fft.ifft2(np.fft.fftshift(filtered_fft2d)))

        if (in_place):
            self.data = filtered_data
                                
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy        
        
    def gaussian_fltr(self, sigma=2, in_place=False):
        '''
        Apply Gaussian filter on data
        
        Usage: * filtering the same data: 
                    Data.gaussian_fltr(in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.gaussian_fltr()
        
        Input: 
            sigma : standard deviation for the Gaussian (default = 2)
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''        
        filtered_data = gaussian_filter(self.data, sigma=sigma)
        
        if (in_place): 
            self.data = filtered_data
            
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy      
        
    def derivative_fltr(self, mode="same", in_place=False):
        '''
        Apply first derivative filter on data
        
        Usage: * filtering the same data: 
                    Data.derivative_fltr(in_place=True)
               * Getting the filterd data:
                    filtered_Data = Data.derivative_fltr()
        
        Input: 
            mode: Convolution mode (default = 'same')
            in_place: Apply filter on the data itself (True) or return a class deep copy with the filtered data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the filtered data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''        
        LE = [[-1*0.5,1*0.5],[-1*0.5,1*0.5]]
        LN = [[1*0.5,1*0.5],[-1*0.5,-1*0.5]]
        
        dGdx = signal.convolve2d(self.data,LE,mode=mode)
        dGdy = signal.convolve2d(self.data,LN,mode=mode)
        
        filtered_data = (dGdx**2) + (dGdy**2) 
        
        if (in_place): 
            self.data = filtered_data
            
        else: 
            copy = self.copy()
            copy.data = filtered_data

            return copy      
                        
    def standarize(self, in_place=False):
        '''
        Standarize data with zero mean and unit standard deviation
        
        Usage: * Standarizing the same data: 
                    Data.standarize(in_place=True)
               * Getting the standarized data:
                    standarized_Data = Data.standarize()
        
        Input: 
            in_place: Standarize the data itself or return a class deep copy with the standarized data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the standarized data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        if (in_place):
            self.data = preprocessing.scale(self.data)
        else: 
            standarized_data = preprocessing.scale(self.data) 
            
            copy = self.copy()
            copy.data = standarized_data

            return copy
        
    def interpolate(self, dt, dx, kind='cubic', in_place=False): 
        '''
        Interpolate data set with new spacing
        
        Usage: * Interpolating the same data: 
                    Data.interpolate(dt, dx, in_place=True)
               * Getting the filterd data:
                    interpolated_Data = Data.interpolate(dt, dx)
        
        Input: 
            dt : New time sampling rate 
            dx : New spacing
            kind : Interpolation method (default = 'cubic')
            in_place: Interpolate the data itself (True) or return a class deep copy with the interpolated data (default = False)
            
        Output:
            in_place = True: None 
            in_place = False: class deep copy with the interpolated data

        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''
        time = self.time;  channel = self.channel
        f = interpolate.interp2d(channel, time, self.data, kind=kind)

        new_time  = np.arange(0, self.time[-1], dt)
        new_depth = np.arange(0, self.depth[-1], dx)

        new_data = f(new_time, new_depth)

        if (in_place): 
            self.freq = 1/dt
            self.k = 1/dx
            self.data = new_data 

        else: 

            copy = self.copy()
            copy.freq = 1/dt
            copy.k = 1/dx
            copy.data = new_data

            return copy

                
    def plt_waterfall(self, use_limits=False, xlim=[0,0], ylim=[0,0], clim=[0,0], h=5, w=15, cmap="gray", aspect='auto'): 
        '''
        Plot (matplotlib) a simple waterfall plot of the data
        
        Usage: * Using default limits: 
                Data.plt_waterfall()
               * Using customized limits
                Data.plt_waterfall(use_limits=True, xlim=[2,8], ylim=[1100,300], clim=[-20,20])
        
        Input: 
            use_limits: Use default limits (False which is the default) or customized limits (True)
            xlim: x-limits pair (default = boundaries)
            ylim: y-limits pair (default = boundaries)
            clim (vmin,vmax) pair for plt.imshow (default = [-30,30])
            h: Plot height (default = 5) 
            w: Plot width (default = 15)
            cmap: color style (default = 'jet')
            aspect: aspect ratio of the axes (default = 'auto')
            
        Output:
            None.  
            
        Written by Ahmed Alharbi, itsAhmed88@gmail.com, 12/2020
        '''   
        if (use_limits): 
        
            plt.figure(figsize=(w,h)) 
            plt.imshow(self.data.T, extent=[self.time[0],self.time[-1], self.channel[-1], self.channel[0]], cmap=cmap, aspect=aspect)
            plt.colorbar(label = 'Amplitude')
            plt.ylabel('Depth (m)'); plt.xlabel('Time (s)')
            plt.clim(clim)
            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.tight_layout()
            plt.show()
            
        else: 
            
            xlim=[self.time[0],self.time[-1]]; ylim=[self.channel[-1], self.channel[0]]; clim=np.array([-1,1])*30
            plt.figure(figsize=(w,h)) 
            plt.imshow(self.data.T, extent=[self.time[0],self.time[-1], self.channel[-1], self.channel[0]], cmap=cmap, aspect=aspect)
            plt.colorbar(label='Amplitude')
            plt.ylabel('Depth (m)'); plt.xlabel('Time (s)')
            plt.clim(clim)
            plt.xlim(xlim)
            plt.ylim(ylim)

            plt.tight_layout()
            plt.show()