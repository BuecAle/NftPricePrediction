import Functions_Atomic
import parameter
import warnings
warnings.filterwarnings('ignore')

# Data collection from Atomichub
Functions_Atomic.atomic_data_collection(parameter.Collection_Atomic.start_date, parameter.Collection_Atomic.end_date)