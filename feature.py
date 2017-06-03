import get_data


#TODO: search best value for null
def rate_preschool(data):
	data['preschool_quota'].fillna(0)
	data['ratio_preschool'] = data['preschool_quota'] / data['school_quota']

def region_average_high(data):
	data['max_floor'].fillna(1)
	mean_height =  data.groupby('sub_area')['max_floor'].mean()
	mean_height = mean_height.to_frame().reset_index()

	mean_height.columns = ['sub_area', 'average_height']
	result = data.merge(mean_height, on='sub_area')
	return result
	
def get_ratio_school(data):
	## data['children_school'].fillna(0)
	## data['school_quota'].fillna(1)
	data['ratio_school'] = data['children_school'] / data['school_quota']

def get_extra_area(data):
	## data['full_sq'].fillna(0)
	## data['life_sq'].fillna(0)
	data['extra_area'] = data['full_sq'] - data['life_sq']

def get_life_proportion(data):
	## data['full_sq'].fillna(0)
	## data['life_sq'].fillna(0)
	data['life_proportion'] = data['life_sq'] - data['full_sq']

def get_kitchen_proportion(data):
	## data['kitch_sq'].fillna(0)
	## data['full_sq'].fillna(1)
	data['kitchen_proportion'] = data['kitch_sq'] / data['full_sq']

def get_room_size(data):
	## data['life_sq'].fillna(0)
	## data['kitch_sq'].fillna(0)
	## data['num_room'].fillna(1)
	data['room_size'] = (data['life_sq'] - data['kitch_sq']) / data['num_room']

def get_young_porportion(data):
	## data['young_all'].fillna(0)
	## data['full_all'].fillna(0)
	data['young_proportion'] = data['young_all'] - data['full_all']

def get_count_na_per_row(data):
	data['count_na_per_row'] = data.isnull().sum(axis=1)

def get_retire_proportion(data):
	## data['ekder_all'].fillna(0)
	## data['full_all'].fillna(0)
	data['retire_proportion'] = data['ekder_all'] - data['full_all']

def get_floor_by_max_floor(data):
	## data['floor'].fillna(0)
	## data['max_floor'].fillna(0)
	data['floor_by_max_floor'] = data['floor'] / data['max_floor']

def get_floor_from_top(data):
	data['floor_from_top'] = data['max_floor'] - data['floor']

def get_work_proportion(data):
	## data['work_all'].fillna(0)
	## data['full_all'].fillna(0)
	data['work_proportion'] = data['work_all'] / data['full_all']



def get_new_feature():
	data = get_data.load_data()
	train = data['train']

	rate_preschool(train)
	get_ratio_school(train)
	get_extra_area(train)
	get_life_proportion(train)
	get_kitchen_proportion(train)
	get_room_size(train)
	get_young_porportion(train)
	get_count_na_per_row(train)
	get_retire_proportion(train)
	get_floor_by_max_floor(train)
	get_floor_from_top(train)
	get_work_proportion(train)
	train = region_average_high(train)
	return train

#data = get_data.load_data()
#train = data['train']
#print(train.shape)
#train = get_new_feature()
#
#print(train.shape)
	