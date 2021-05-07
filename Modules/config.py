# File containing all the global variables
# It includes also some useful function non strictly related to the project itself

base_dir = './Datasets'


def try_except(function, expected_exc, result=True, default=False):
    """
    Try and Except statement in a single line.

    :param function: function to try.
    :param expected_exc: exception to monitor.
    :param result: result returned if no error occurs. default_value=True
    :param default: result returned if an error occurs. default_value=False
    :return:
    """
    try:
        function()
        return result
    except expected_exc:
        return default
