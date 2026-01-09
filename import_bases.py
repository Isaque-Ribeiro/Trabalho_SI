import kagglehub
from kagglehub import KaggleDatasetAdapter

# Arquivo a ser usado
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "jessicali9530/lfw-dataset",
  file_path,
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())