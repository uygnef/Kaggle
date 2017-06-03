import get_data


data = get_data.load_data()

train = data['train']

#TODO: search best value for null
def rate_preschool(data):
    data['preschool_quota'].fillna(0)
    data['ratio_preschool'] = data['preschool_quota'] / data['school_quota']
    data['age_at_sale'] = data['']

print(train.head())