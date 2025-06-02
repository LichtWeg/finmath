import pandas as pd
import matplotlib.pyplot as plt
import pprint

# Load the CSV
file_path = 'src/finmath/Jan2May2025.csv'  # Update this with your real path
df = pd.read_csv(file_path, sep=';', decimal=',')

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert date & amount
df['booking_date'] = pd.to_datetime(df['booking_date'], format='%d.%m.%y')
df['amount'] = df['amount'].astype(float)

# Basic overview
print("### Head:\n", df.head())
print("\n### Info:")
df.info()
print("\n### Summary Stats:\n", df.describe())

# Transaction type analysis
tx_summary = df.groupby('transaction_type')['amount'].agg(['count', 'sum'])
print("\n### Transaction Type Summary:\n", tx_summary)

# Top vendors (keyword extraction from booking text)
df['vendor'] = df['booking_text'].str.extract(r'^(.*?)(//|,|End-To-End|SEPA|$)')[0].str.strip()
vendor_summary = df.groupby('vendor')['amount'].sum().sort_values()
print("\n### Top Vendors:\n", vendor_summary.tail(10))


# Group and sum amounts by category
category_sums = df.groupby('category')['amount'].sum().abs()
total_sum = category_sums.sum()

# Custom label function
def make_autopct(values):
    def my_autopct(pct):
        val = pct / 100.0 * sum(values)
        return f'{pct:.1f}%\n(€{val:,.2f})'
    return my_autopct


# Group and sum the absolute amounts by category
category_sums = df.groupby('category')['amount'].sum().abs()
total_sum = category_sums.sum()

# Build the dictionary
category_summary = {
    category: {
        'percent': round((amount / total_sum) * 100, 1),
        'amount': f'{round(amount, 2)} EUR'
    }
    for category, amount in category_sums.items()
}

# Print the result
pprint.pprint(category_summary)


# Plot pie chart
plt.figure(figsize=(9, 9))
plt.pie(category_sums, 
        labels=category_sums.index,
        autopct=make_autopct(category_sums), 
        startangle=140, 
        colors=plt.cm.Set3.colors)
plt.title(f'Transaction Amount by Category\nTotal: €{total_sum:,.2f}')
plt.axis('equal')
plt.tight_layout()
plt.show()

