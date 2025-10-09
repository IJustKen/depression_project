def feat_eng(df):
  # combining satisfaction from studies and job both and getting average
    df['Average_Satisfaction'] = (df['Study Satisfaction'] + df['Job Satisfaction']) / 2
  
  # this measures how resilient one is to pressure, we are assuming the hypothesis that if your CGPA is high
  # while having high pressure and studying a lot it means they are more resilient to depression.
  # we could assume the opposite for someone prone to depression.
    df['Resilience'] = df['CGPA'] * (df['Academic Pressure'] + df['Work Pressure']) * df['Work/Study Hours']
  
  # this feature measures the balance between stress and satisfaction. A high value might indicate an unhealthy balance.
    df['Pressure_Satisfaction_Ratio'] = df['Total_Pressure'] / (df['Average_Satisfaction'] + 0.01)
  # adding a small number to the denominator to avoid division by zero

    return df





