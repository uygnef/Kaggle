import get_data


data = get_data.load_data()

train = data['train']

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

train = region_average_high(train)

print(train.shape)


