import pandas as pd
import json

def xlsx_to_json(xlsx_file, json_file):
    # 讀取Excel檔案中的每個活頁簿並轉換為JSON格式
    data = {}
    with pd.ExcelFile(xlsx_file) as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # 檢查DataFrame是否為空
            if not df.empty:
                records = df.where((pd.notnull(df)), None).to_dict(orient='records')
                data[sheet_name] = [record for record in records if any(record.values())]
        
    # 將JSON資料寫入檔案，並確保中文正常顯示
    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)

# 指定要讀取的Excel檔案位置和要寫入的JSON檔案位置
xlsx_file = 'D:/Desktop/test.xlsx'
json_file = 'D:/Desktop/test.json'

# 呼叫函式將Excel轉換為JSON
xlsx_to_json(xlsx_file, json_file)
