#!/bin/bash

url='http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
name='online_retail.xlsx'
out='online_retail.pkl'

code="
import pandas, pickle

print('Loading $name...')
df = pandas.read_excel('./$name')

print('Formatting...')
keys = ['InvoiceNo', 'StockCode', 'Description']
transactions, descriptions = {}, {}
for invoice, code, description in df.get(keys).as_matrix():
    invoice, code, description = str(invoice), str(code), str(description)
    if description == 'nan':
        continue
    descriptions[code] = description.strip().title()
    if invoice not in transactions:
        transactions[invoice] = {code}
    else:
        transactions[invoice].add(code)
data = {
    'descriptions': descriptions,
    'transactions': transactions
}

print('Saving to $out...')
pickle.dump(data, open('./$out', 'wb'))
"

[ -f "$name" ] \
    || ( echo "Downloading $url..." \
    && curl "$url" > "$name" \
    && echo "OK" )

[ -f "$name" -a ! -f "$out" ] \
    && echo "Converting $name to pickle: $out..." \
    && python3 -c "$code" \
    && echo "OK"
