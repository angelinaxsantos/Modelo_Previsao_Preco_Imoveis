import pandas as pd

dataset_1 = pd.read_csv('Datasets/kc_house_data.csv', low_memory=False)
dataset_2 = pd.read_csv('Datasets/portland_housing.csv', low_memory=False)

dataset_1 = dataset_1.rename(columns={'date': 'dateSold'})
dataset_1 = dataset_1.rename(columns={'sqft_living': 'livingArea'})
dataset_1 = dataset_1.rename(columns={'sqft_lot': 'lotSize'})
dataset_1 = dataset_1.rename(columns={'yr_built': 'yearBuilt'})
dataset_1 = dataset_1.rename(columns={'lat': 'latitude'})
dataset_1 = dataset_1.rename(columns={'long': 'longitude'})
dataset_2 = dataset_2.rename(columns={'address/zipcode': 'zipcode'})
dataset_2 = dataset_2.rename(columns={'address/city': 'city'})

dataset_1 = dataset_1.drop(columns=['id', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_renovated', 'sqft_living15', 'sqft_lot15'])
dataset_2 = dataset_2.drop(columns=['city', 'abbreviatedAddress', 'brokerageName', 'daysOnZillow', 'description', 'favoriteCount', 'homeStatus', 'homeType', 'lastSoldPrice', 'pageViewCount', 'postingContact/name', 'priceHistory/0/attributeSource/infoString1', 'priceHistory/0/attributeSource/infoString2', 'priceHistory/0/attributeSource/infoString3', 'priceHistory/0/buyerAgent/name', 'priceHistory/0/buyerAgent/photo/url', 'priceHistory/0/buyerAgent/profileUrl', 'priceHistory/0/date', 'priceHistory/0/event', 'priceHistory/0/postingIsRental', 'priceHistory/0/price', 'priceHistory/0/priceChangeRate', 'priceHistory/0/pricePerSquareFoot', 'priceHistory/0/sellerAgent/name', 'priceHistory/0/sellerAgent/photo/url', 'priceHistory/0/sellerAgent/profileUrl', 'priceHistory/0/showCountyLink', 'priceHistory/0/source', 'priceHistory/0/time', 'priceHistory/1/attributeSource/infoString1', 'priceHistory/1/attributeSource/infoString2', 'priceHistory/1/attributeSource/infoString3', 'priceHistory/1/date', 'priceHistory/1/event', 'priceHistory/1/postingIsRental', 'priceHistory/1/price', 'priceHistory/1/priceChangeRate', 'priceHistory/1/pricePerSquareFoot', 'priceHistory/1/showCountyLink', 'priceHistory/1/source', 'priceHistory/1/time', 'priceHistory/2/attributeSource/infoString1', 'priceHistory/2/attributeSource/infoString2', 'priceHistory/2/attributeSource/infoString3', 'priceHistory/2/date', 'priceHistory/2/event', 'priceHistory/2/postingIsRental', 'priceHistory/2/price', 'priceHistory/2/priceChangeRate', 'priceHistory/2/pricePerSquareFoot', 'priceHistory/2/showCountyLink', 'priceHistory/2/source', 'priceHistory/2/time', 'priceHistory/3/attributeSource/infoString2', 'priceHistory/3/date', 'priceHistory/3/event', 'priceHistory/3/postingIsRental', 'priceHistory/3/price', 'priceHistory/3/priceChangeRate', 'priceHistory/3/pricePerSquareFoot', 'priceHistory/3/showCountyLink', 'priceHistory/3/source', 'priceHistory/3/time', 'priceHistory/4/attributeSource/infoString2', 'priceHistory/4/date', 'priceHistory/4/event', 'priceHistory/4/postingIsRental', 'priceHistory/4/price', 'priceHistory/4/priceChangeRate', 'priceHistory/4/pricePerSquareFoot', 'priceHistory/4/showCountyLink', 'priceHistory/4/source', 'priceHistory/4/time', 'priceHistory/5/attributeSource/infoString2', 'priceHistory/5/date', 'priceHistory/5/event', 'priceHistory/5/postingIsRental', 'priceHistory/5/price', 'priceHistory/5/priceChangeRate', 'priceHistory/5/pricePerSquareFoot', 'priceHistory/5/showCountyLink', 'priceHistory/5/source', 'priceHistory/5/time', 'priceHistory/6/attributeSource/infoString2', 'priceHistory/6/date', 'priceHistory/6/event', 'priceHistory/6/postingIsRental', 'priceHistory/6/price', 'priceHistory/6/priceChangeRate', 'priceHistory/6/pricePerSquareFoot', 'priceHistory/6/showCountyLink', 'priceHistory/6/source', 'priceHistory/6/time', 'priceHistory/7/attributeSource/infoString2', 'priceHistory/7/date', 'priceHistory/7/event', 'priceHistory/7/postingIsRental', 'priceHistory/7/price', 'priceHistory/7/priceChangeRate', 'priceHistory/7/pricePerSquareFoot', 'priceHistory/7/showCountyLink', 'priceHistory/7/source', 'priceHistory/7/time', 'priceHistory/8/attributeSource/infoString2', 'priceHistory/8/date', 'priceHistory/8/event', 'priceHistory/8/postingIsRental', 'priceHistory/8/price', 'priceHistory/8/priceChangeRate', 'priceHistory/8/pricePerSquareFoot', 'priceHistory/8/showCountyLink', 'priceHistory/8/source', 'priceHistory/8/time', 'priceHistory/9/attributeSource/infoString2', 'priceHistory/9/date', 'priceHistory/9/event', 'priceHistory/9/postingIsRental', 'priceHistory/9/price', 'priceHistory/9/priceChangeRate', 'priceHistory/9/pricePerSquareFoot', 'priceHistory/9/showCountyLink', 'priceHistory/9/source', 'priceHistory/9/time', 'priceHistory/10/attributeSource/infoString2', 'priceHistory/10/date', 'priceHistory/10/event', 'priceHistory/10/postingIsRental', 'priceHistory/10/price', 'priceHistory/10/priceChangeRate', 'priceHistory/10/showCountyLink', 'priceHistory/10/source', 'priceHistory/10/time', 'propertyTaxRate', 'rentZestimate', 'rentalApplicationsAcceptedType', 'resoFacts/accessibilityFeatures/0', 'resoFacts/accessibilityFeatures/1', 'resoFacts/appliances/0', 'resoFacts/appliances/1', 'resoFacts/appliances/2', 'resoFacts/appliances/3', 'resoFacts/appliances/4', 'resoFacts/appliances/5', 'resoFacts/appliances/6', 'resoFacts/appliances/7', 'resoFacts/appliances/8', 'resoFacts/architecturalStyle', 'resoFacts/associationAmenities/0', 'resoFacts/associationFee', 'resoFacts/atAGlanceFacts/0/factLabel', 'resoFacts/atAGlanceFacts/0/factValue', 'resoFacts/atAGlanceFacts/1/factLabel', 'resoFacts/atAGlanceFacts/1/factValue', 'resoFacts/atAGlanceFacts/2/factLabel', 'resoFacts/atAGlanceFacts/2/factValue', 'resoFacts/atAGlanceFacts/3/factLabel', 'resoFacts/atAGlanceFacts/3/factValue', 'resoFacts/atAGlanceFacts/4/factLabel', 'resoFacts/atAGlanceFacts/4/factValue', 'resoFacts/atAGlanceFacts/5/factLabel', 'resoFacts/atAGlanceFacts/5/factValue', 'resoFacts/atAGlanceFacts/6/factLabel', 'resoFacts/atAGlanceFacts/6/factValue', 'resoFacts/atAGlanceFacts/7/factLabel', 'resoFacts/basement', 'resoFacts/constructionMaterials/0', 'resoFacts/constructionMaterials/1', 'resoFacts/cooling/0', 'resoFacts/doorFeatures/0', 'resoFacts/exteriorFeatures/0', 'resoFacts/exteriorFeatures/1', 'resoFacts/exteriorFeatures/2', 'resoFacts/fencing', 'resoFacts/fireplaceFeatures/0', 'resoFacts/fireplaces', 'resoFacts/flooring/0', 'resoFacts/flooring/1', 'resoFacts/flooring/2', 'resoFacts/flooring/3', 'resoFacts/flooring/4', 'resoFacts/foundationDetails/0', 'resoFacts/furnished', 'resoFacts/gas/0', 'resoFacts/hasAdditionalParcels', 'resoFacts/hasAssociation', 'resoFacts/hasAttachedGarage', 'resoFacts/hasAttachedProperty', 'resoFacts/hasCarport', 'resoFacts/hasCooling', 'resoFacts/hasFireplace', 'resoFacts/hasGarage', 'resoFacts/hasHeating', 'resoFacts/hasHomeWarranty', 'resoFacts/hasLandLease', 'resoFacts/hasOpenParking', 'resoFacts/hasPetsAllowed', 'resoFacts/hasSpa', 'resoFacts/hasView', 'resoFacts/heating/0', 'resoFacts/heating/1', 'resoFacts/homeType', 'resoFacts/interiorFeatures/0', 'resoFacts/interiorFeatures/1', 'resoFacts/interiorFeatures/2', 'resoFacts/interiorFeatures/3', 'resoFacts/interiorFeatures/4', 'resoFacts/interiorFeatures/5', 'resoFacts/interiorFeatures/6', 'resoFacts/interiorFeatures/7', 'resoFacts/interiorFeatures/8', 'resoFacts/interiorFeatures/9', 'resoFacts/interiorFeatures/10', 'resoFacts/interiorFeatures/11', 'resoFacts/isNewConstruction', 'resoFacts/laundryFeatures/0', 'resoFacts/laundryFeatures/1', 'resoFacts/lotFeatures/0', 'resoFacts/lotFeatures/1', 'resoFacts/onMarketDate', 'resoFacts/otherStructures/0', 'resoFacts/parcelNumber', 'resoFacts/parking', 'resoFacts/parkingFeatures/0', 'resoFacts/parkingFeatures/1', 'resoFacts/parkingFeatures/2', 'resoFacts/parkingFeatures/3', 'resoFacts/patioAndPorchFeatures/0', 'resoFacts/patioAndPorchFeatures/1', 'resoFacts/propertyCondition', 'resoFacts/propertySubType/0', 'resoFacts/roofType', 'resoFacts/rooms/0/roomArea', 'resoFacts/rooms/0/roomFeatures/0', 'resoFacts/rooms/0/roomFeatures/1', 'resoFacts/rooms/0/roomFeatures/2', 'resoFacts/rooms/0/roomLength', 'resoFacts/rooms/0/roomLevel', 'resoFacts/rooms/0/roomType', 'resoFacts/rooms/0/roomWidth', 'resoFacts/rooms/1/roomArea', 'resoFacts/rooms/1/roomFeatures/0', 'resoFacts/rooms/1/roomFeatures/1', 'resoFacts/rooms/1/roomLength', 'resoFacts/rooms/1/roomLevel', 'resoFacts/rooms/1/roomType', 'resoFacts/rooms/1/roomWidth', 'resoFacts/rooms/2/roomArea', 'resoFacts/rooms/2/roomFeatures/0', 'resoFacts/rooms/2/roomFeatures/1', 'resoFacts/rooms/2/roomLength', 'resoFacts/rooms/2/roomLevel', 'resoFacts/rooms/2/roomType', 'resoFacts/rooms/2/roomWidth', 'resoFacts/rooms/3/roomArea', 'resoFacts/rooms/3/roomFeatures/0', 'resoFacts/rooms/3/roomFeatures/1', 'resoFacts/rooms/3/roomLength', 'resoFacts/rooms/3/roomLevel', 'resoFacts/rooms/3/roomType', 'resoFacts/rooms/3/roomWidth', 'resoFacts/rooms/4/roomArea', 'resoFacts/rooms/4/roomFeatures/0', 'resoFacts/rooms/4/roomFeatures/1', 'resoFacts/rooms/4/roomFeatures/2', 'resoFacts/rooms/4/roomLength', 'resoFacts/rooms/4/roomLevel', 'resoFacts/rooms/4/roomType', 'resoFacts/rooms/4/roomWidth', 'resoFacts/rooms/5/roomArea', 'resoFacts/rooms/5/roomFeatures/0', 'resoFacts/rooms/5/roomFeatures/1', 'resoFacts/rooms/5/roomFeatures/2', 'resoFacts/rooms/5/roomLength', 'resoFacts/rooms/5/roomLevel', 'resoFacts/rooms/5/roomType', 'resoFacts/rooms/5/roomWidth', 'resoFacts/rooms/6/roomArea', 'resoFacts/rooms/6/roomFeatures/0', 'resoFacts/rooms/6/roomFeatures/1', 'resoFacts/rooms/6/roomFeatures/2', 'resoFacts/rooms/6/roomLength', 'resoFacts/rooms/6/roomLevel', 'resoFacts/rooms/6/roomType', 'resoFacts/rooms/6/roomWidth', 'resoFacts/rooms/7/roomLevel', 'resoFacts/rooms/7/roomType', 'resoFacts/securityFeatures/0', 'resoFacts/sewer/0', 'resoFacts/stories', 'resoFacts/subdivisionName', 'resoFacts/view/0', 'resoFacts/waterSource/0', 'resoFacts/waterViewYN', 'resoFacts/windowFeatures/0', 'resoFacts/windowFeatures/1', 'restimateHighPercent', 'restimateLowPercent', 'schools/0/distance', 'schools/0/level', 'schools/0/link', 'schools/0/name', 'schools/0/rating', 'schools/0/size', 'schools/0/studentsPerTeacher', 'schools/0/totalCount', 'schools/1/distance', 'schools/1/level', 'schools/1/link', 'schools/1/name', 'schools/1/rating', 'schools/1/size', 'schools/1/studentsPerTeacher', 'schools/1/totalCount', 'schools/2/distance', 'schools/2/level', 'schools/2/link', 'schools/2/name', 'schools/2/rating', 'schools/2/size', 'schools/2/studentsPerTeacher', 'schools/2/totalCount', 'solarPotential/buildFactor', 'solarPotential/climateFactor', 'solarPotential/electricityFactor', 'solarPotential/solarFactor', 'solarPotential/sunScore', 'taxAssessedValue', 'taxAssessedYear', 'taxHistory/0/taxIncreaseRate', 'taxHistory/0/taxPaid', 'taxHistory/0/time', 'taxHistory/0/value', 'taxHistory/0/valueIncreaseRate', 'url', 'zestimate', 'zestimateHighPercent', 'zestimateLowPercent', 'zpid'])

dataset_2['dateSold'] = pd.to_datetime(dataset_2['dateSold'], unit='ms')

df = pd.concat([dataset_1, dataset_2], ignore_index=True)

df = df.drop(columns=['latitude', 'longitude', 'zipcode'])

file ="Datasets/Dataset_01.csv"
df.to_csv(file, index=False)