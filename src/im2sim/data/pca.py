
class PCA:
    """
    Principal Component Analysis 

    This class can be used for PCA-related operations such as computing PCs and saving PC projection matrices

    Parameters
    ----------
    data: np.ndarraye
        data to PCA
    axis: int
        axis to conduct the PCA over

    Attributes
    ----------
    S: np.ndarray
        PC matrix
    V: np.ndarray
        Variance explained by each PC
    """

    def __init__(self, data, axis=-1):
        '''
        takes the data and axis, computes the PCs matrix and stores in object attributes
        '''
        raise NotImplementedError

    def save(self):
        '''saves the PC data'''
        raise NotImplementedError

    def load(self, fname):
        '''loads saved PC data'''
        raise NotImplementedError

    def forward_transform(data):
        '''forward transform'''
        raise NotImplementedError
    
    def inverse_transform(data):
        '''inverse transform'''
        raise NotImplementedError

    def forward_transform_tf(data):
        '''forward transform'''
        raise NotImplementedError
    
    def inverse_transform_tf(data):
        '''inverse transform'''
        raise NotImplementedError