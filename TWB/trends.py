import pandas as pd
from pytrends.request import TrendReq
import os  
import numpy as np
import datetime as dt
import glob
import warnings
warnings.filterwarnings('ignore')

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.io import output_notebook
from datetime import datetime as dt
from bokeh.models import DatetimeTickFormatter

class Trends(object):
    """
    Returns information from google trends cached data.
    """
    def __init__(self, country, *args, **kwargs):
        """
        Get the information for a specific country
        
        parameters:
        -----------
        country in full letters (ex. Greece),
        
        optional:
        timestamp: '2016-05',
        window (int)
        country_for_comparison
        """
        self._country = country
        self._timestamp = kwargs.get('timestamp', None)
        self._window = kwargs.get('window', None)
        self._country_for_comparison = kwargs.get('country_for_comparison', None)
        self._validation_df = kwargs.get('validation_df', None)
        
    
    #edef
    
    @property
    def get_maximum_occurences_on_country_by_month_with_window(self):
        """
        Returns the max occurences on a particular country
        """
        timestamp3_df = pd.DataFrame(columns=('timestamp', 'occurences', 'reason'))
        for name in glob.glob('Data/trends/'+self._country+'*'):
            country_term2 = pd.read_pickle(name)
            if not country_term2.empty:
                #this dataframe groups by month
                df2 = country_term2.groupby(pd.TimeGrouper('M')).max()
                #print(df2)

                #find the indexes of the date index(column) that refers to the monthly timestamp

                #get column name that matches the country name (in order to avoid the reference country)
                spike_cols2 = [col for col in country_term2.columns if self._country in col]

                cc2 = country_term2.loc[self._timestamp][spike_cols2]
                #print(cc2)
                timestamp3_df = timestamp3_df.append({'timestamp': self._timestamp, 'occurences': cc2.iloc[0][0].astype('float'), 'reason': spike_cols2[0]}, ignore_index=True)
                #print(timestamp3_df)
        #timestamp3_df.occurences = timestamp3_df.occurences.astype('float64')
        #max_index = timestamp3_df.occurences.idxmax()
        #print(timestamp3_df['occurences'].max())

        max_index = (timestamp3_df.ix[timestamp3_df.occurences.idxmax()].name)
        max_value  = timestamp3_df.ix[timestamp3_df.occurences.idxmax()].occurences
        if not (max_index-self._window < 0):
        
            #go back to all the dataframes and find the one that matches the max and return the time window on that

            max_reason = timestamp3_df.iloc[max_index].reason
            max_reason2 = max_reason.replace(" ", "_")
            #print(max_reason2)
            for name in glob.glob('Data/trends/'+max_reason2+'*'):
                country_term3 = pd.read_pickle(name)
                #print(country_term3)
                if not country_term3.empty:
                    #print(country_term3)
                    my_old_index = country_term3.loc[(country_term3[max_reason] == max_value) & (country_term3.index == self._timestamp)]



                    #my_index = (country_term3.loc[country_term3[max_reason] == max_value && country_term3.index.match(timestamp)].index)
                    if not my_old_index.empty:
                        #print(my_old_index.index.get_indexer)
                        my_new_index = (country_term3.index.get_loc(my_old_index.index.values[0]))
                        #print(my_new_index)
                        output = country_term3.iloc[my_new_index - self._window: my_new_index+self._window]
                        return(output)
                        
    @property
    def download_trends_for_a_country_general(self):
        """
        Download general information for a particular country
        """
        
        
        if not (os.path.isdir("Data/trends/countries")):
            os.mkdir('Data/trends/countries/')
            
        #check if country file already exists
        #print('Data/trends/countries/'+self._country+'.pkl')
        if not (os.path.exists("Data/trends/countries/"+self._country+'.pkl')):
            pytrend = TrendReq(hl='en-US', tz=360)
            pytrend.build_payload([self._country], timeframe='all')
            interest_over_time_df = pytrend.interest_over_time()
            print("Country data do not exist, downloading now...")
            interest_over_time_df.to_pickle('Data/trends/countries/'+str(self._country)+'.pkl')
            print('Country information downloaded')
            #region_df = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=False, inc_geo_code=False)
        else:
            print("You are lucky, country already downloaded")
    
    @property
    def plot_general_information_for_a_country(self):
        """
        Plot trends for only only country without any tag
        """
        
        
        df2 = pd.read_pickle('Data/trends/countries/'+ self._country+'.pkl')

        output_notebook()
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        TOOLTIPS = [
            ('month', '@x{datetime}'),
            ('Total searches', '@y'),
        ]

        p = figure(title="Monthly-wise total search of term - "+str(self._country), y_axis_type="linear", plot_height = 400,
                   tools = TOOLS, tooltips = TOOLTIPS, plot_width = 800)


        p.xaxis.axis_label = 'Month'
        p.yaxis.axis_label = 'Total search'
        #p.circle(2010, temp_df.IncidntNum.min(), size = 10, color = 'red')

        p.line(df2.index, df2[str(self._country)],line_color="purple", line_width = 3)

        p.xaxis.formatter=DatetimeTickFormatter(

                days=["%d %B %Y"],
                months=["%d %B %Y"],
                years=["%d %B %Y"],
            )
        # output_file("line_chart.html", title="Line Chart")
        show(p)

        
    @property
    def download_trends_for_compared_countries_general(self):
        comp_list = [self._country, self._country_for_comparison]
        if not (os.path.isdir("Data/trends/countries")):
                    os.mkdir('Data/trends/countries/')

                #check if country file already exists
                #print('Data/trends/countries/'+self._country+'.pkl')
        if not (os.path.exists("Data/trends/countries/"+self._country+'_compared'+'.pkl')):
            pytrend = TrendReq(hl='en-US', tz=360)
            pytrend.build_payload(comp_list, timeframe='all')
            interest_over_time_df = pytrend.interest_over_time()
            print("Country data do not exist, downloading now...")
            interest_over_time_df.to_pickle('Data/trends/countries/'+str(comp_list[0])+'_compared'+'.pkl')
            print('Country information downloaded')
            #region_df = pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=False, inc_geo_code=False)
        else:
            print("You are lucky, country already downloaded")

    
    
    @property
    def plot_general_information_for_compared_countries_general(self):
        """
        Plot trends for compared countries without any tag
        Needs to get an additional country as kwarg
        """
        
        df2 = pd.read_pickle('Data/trends/countries/'+ self._country+'_compared'+'.pkl')

        output_notebook()
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        TOOLTIPS = [
            ('month', '@x{datetime}'),
            ('Total searches', '@y'),
        ]

        p = figure(title="Monthly-wise total search of term - "+str(self._country) +'-'+str(self._country_for_comparison), y_axis_type="linear", plot_height = 400,
                   tools = TOOLS, tooltips = TOOLTIPS, plot_width = 800)


        p.xaxis.axis_label = 'Month'
        p.yaxis.axis_label = 'Total search'
        #p.circle(2010, temp_df.IncidntNum.min(), size = 10, color = 'red')

        p.line(df2.index, df2[str(self._country)],line_color="purple", line_width = 3)
        p.line(df2.index, df2[str(self._country_for_comparison)],line_color="purple", line_width = 3)

        p.xaxis.formatter=DatetimeTickFormatter(

                days=["%d %B %Y"],
                months=["%d %B %Y"],
                years=["%d %B %Y"],
            )
        # output_file("line_chart.html", title="Line Chart")
        show(p)
        
    @property
    def plot_validation_comparison(self):
        #first check if general country information exists
        #if not download
        
        if not (os.path.exists("Data/trends/countries/"+self._country+'.pkl')):
            print("Original data for country to compare do not exist, download now")
            self.download_trends_for_a_country_general
        #edif
        output_notebook()
        TOOLS = 'save,pan,box_zoom,reset,wheel_zoom,hover'
        TOOLTIPS = [
            ('month', '@x{datetime}'),
            ('Total searches', '@y'),
        ]
        
        
        df2 = pd.read_pickle('Data/trends/countries/'+ self._country+'.pkl')

        p = figure(title="Compare data for "+self._country, y_axis_type="linear", plot_height = 400,
                   tools = TOOLS, tooltips = TOOLTIPS, plot_width = 800)


        p.xaxis.axis_label = 'Month'
        p.yaxis.axis_label = 'Total search'
        #p.circle(2010, temp_df.IncidntNum.min(), size = 10, color = 'red')

        p.line(self._validation_df.index, self._validation_df.values,line_color="purple", line_width = 3)
        p.line(df2.index, df2[str(self._country)],line_color="red", line_width = 3)
        

        p.xaxis.formatter=DatetimeTickFormatter(

                days=["%d %B %Y"],
                months=["%d %B %Y"],
                years=["%d %B %Y"],
            )
        # output_file("line_chart.html", title="Line Chart")
        show(p)
            
            
        