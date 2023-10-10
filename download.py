import numpy as np
import requests
import pandas as pd



def ssb(url, jsonstr):
	# Fetch the data from the API
	response = requests.post(url, data=jsonstr)
	data = response.json()

	# Extract the values and reshape into a 3D array
	values = np.array(data['value']).reshape(*data['size'])
	headers = data['id']

	dims =[data['dimension'][i]['category']['label'] for i in headers]
	dims = [[d[k] for k in d] for d in dims]
	df = pd.DataFrame(values.flatten(), columns=['Data'])
	s = values.shape
	x = np.empty([len(df),len(dims)], dtype='<U100')
	ix = np.arange(len(df)).reshape(s)
	for m in range(len(dims)):
		ix_sw = ix.swapaxes(m,len(dims)-1)
		h = ix_sw.reshape((int(np.prod(s)/s[m]),s[m]))
		x[h,m] = dims[m]

	lables = pd.DataFrame(x, columns = headers)
	df = pd.concat([lables, df],axis = 1)
	return df
