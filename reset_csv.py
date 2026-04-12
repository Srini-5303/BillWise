"""
One-time script to delete the existing GCS CSV and create a fresh one
with the updated schema (Item_Name, Item_Price, Grocery_Category columns).

Run once:
    python reset_csv.py

WARNING: This permanently deletes all existing receipt data in GCS.
"""

from csv_writer import reset_csv

if __name__ == "__main__":
    confirm = input("This will DELETE all existing receipt data. Type 'yes' to continue: ")
    if confirm.strip().lower() == "yes":
        reset_csv()
        print("CSV reset. New headers written to GCS.")
    else:
        print("Aborted.")
