import pandas as pd

# Read the fraud data and extract transcript column
fraud_df = pd.read_csv('../Training_data/fraud_metadata_actual.csv')
fraud_transcripts = fraud_df[['transcript']].copy()
fraud_transcripts['label'] = 'fraud'

# Read the normal data and extract Transcription column
normal_df = pd.read_csv('../Training_data/normal_transcriptions.csv')
normal_transcripts = normal_df[['Transcription']].copy()
normal_transcripts['label'] = 'normal'

# Rename the Transcription column to match the fraud dataset
normal_transcripts = normal_transcripts.rename(columns={'Transcription': 'transcript'})

# Concatenate the two dataframes
combined_df = pd.concat([fraud_transcripts, normal_transcripts], ignore_index=True)

# Save to CSV
combined_df.to_csv('combined_data.csv', index=False)

# Print some information about the combined dataset
print(f"Total number of samples: {len(combined_df)}")
print(f"Number of fraud samples: {len(fraud_transcripts)}")
print(f"Number of normal samples: {len(normal_transcripts)}")
print("\nFirst few rows of combined dataset:")
print(combined_df.head())
