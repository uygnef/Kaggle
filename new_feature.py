import get_data


data = get_data.load_data()

train = data['train']

#TODO: search best value for null
def rate_preschool(data):
    data['preschool_quota'].fillna(0)
    data['ratio_preschool'] = data['preschool_quota'] / data['school_quota']

def region_average_high(data):
    mean_height = train.groupby('sub_area')['max_floor'].mean()
    

print()


