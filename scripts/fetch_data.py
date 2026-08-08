from ucimlrepo import fetch_ucirepo 
import pandas as pd

class FetchData:
    def __init__(self):
        self.data = None
        self.metadata = None

    def fetch_data(self, id):
        data = fetch_ucirepo(id=id)
        self.data = data
        return data

    def _to_dataframe(self, data):
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame([dict(data)])

        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame([dict(data)])

    def get_metadata(self):
        if self.data is not None:
            return self._to_dataframe(self.data.metadata)
        else:
            raise ValueError("Data not fetched yet. Please call fetch_data() first.")
    
    def get_variables(self):
        if self.data is not None:
            return self._to_dataframe(self.data.variables)
        else:
            raise ValueError("Data not fetched yet. Please call fetch_data() first.")

# # fetch dataset 
# daily_demand_forecasting_orders = fetch_ucirepo(id=409) 
  
# # data (as pandas dataframes) 
# X = daily_demand_forecasting_orders.data.features 
# y = daily_demand_forecasting_orders.data.targets 
  
# # metadata 
# print(daily_demand_forecasting_orders.metadata) 
  
# # variable information 
# print(daily_demand_forecasting_orders.variables) 
