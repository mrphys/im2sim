# TODO: Investigate PowerScaling to replace the StandardScaler and DataProc 



class StandardScaler:
    """
    Z-score scaler for node features

    This class can be used to create a z-score/standard scaler for node data.

    Parameters
    ----------
    mean: float, list of floats
        mean of the node feature(s)
    std: float, list of floats
        standard deviation of node feature(s)

    Attributes
    ----------
    mean: float, list of floats
        mean of the node feature(s)
    std: float, list of floats
        standard deviation of node feature(s)
    """

class Normaliser:
    """
    min-max normaliser for node features

    This class can be used to create a 0-1 normaliser for node data.

    Parameters
    ----------
    min: float, list of floats
        min value of the node feature(s)
    max: float, list of floats
        max value of node feature(s)

    Attributes
    ----------
    min: float, list of floats
        min value of the node feature(s)
    max: float, list of floats
        max value of node feature(s)
    """

    
class DataProcessor:
    """
    Node Data Processor for Im2Sim models

    This class provides the basis for building custom data generators for Im2Sim models.

    Parameters
    ----------
    data_gen: function
        Function that yeilds input-output pairs for each da

    Attributes
    ----------
    attr1 : type
        Description of `attr1`.
    attr2 : type
        Description of `attr2`.

    Examples
    --------
    >>> obj = MyClass(param1=10, param2="test")
    >>> obj.method()
    42 """

