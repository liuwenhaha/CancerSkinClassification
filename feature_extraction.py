from radiomics import featureextractor
import pandas as pd

root_dir = "./skin-cancer-mnist-ham10000"
metadata_path = "{}/HAM10000_metadata.csv".format(root_dir)
directory_to_write = "./nrrd"
df = pd.read_csv(metadata_path)[['image_id', 'dx']]
l = len(df)
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('ngtdm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('glszm')


def extract_all_features(extractor, image_name, root_dir, diagnosis):
    image_path = "{0}/{1}/{2}.nrrd".format(root_dir, diagnosis, image_name)
    label_path = "{0}/{1}/{2}-label.nrrd".format(root_dir, diagnosis, image_name)
    return extractor.execute(image_path, label_path)


def clean_features(features):
    toBeDeleted = ['diagnostics_Image-original_Dimensionality', 'diagnostics_Versions_PyRadiomics',
                   'diagnostics_Versions_Numpy',
                   'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet',
                   'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings',
                   'diagnostics_Configuration_EnabledImageTypes', 'diagnostics_Image-original_Hash',
                   'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size',
                   'diagnostics_Mask-original_Hash', 'diagnostics_Mask-original_Spacing',
                   'diagnostics_Mask-original_Size', 'diagnostics_Mask-original_BoundingBox',
                   'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
                   'diagnostics_Mask-original_CenterOfMassIndex', 'diagnostics_Mask-original_CenterOfMass']
    for feature in range(len(toBeDeleted)):
        del (features[toBeDeleted[feature]])
    return features


keys = clean_features(extract_all_features(extractor, "ISIC_0024318", directory_to_write, "df")).keys()
for key in keys:
    df[key] = df.apply(lambda _: '', axis=1)


for index, row in df.iterrows():
    features = clean_features(extract_all_features(extractor, row['image_id'], directory_to_write, row['dx']))
    for feature in features:
        df.at[index, feature] = features[feature]
    print((index + 1)/l)

df_csv = df.to_csv(index=False)
with open("result.csv", "w") as file:
    file.write(df_csv)