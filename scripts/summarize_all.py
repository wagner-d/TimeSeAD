'''
Script to consolidate json results from summarize_exp output to a single Excel file
'''
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Color
import sys
import glob
import json
import logging
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

class SheetTracker:
    CHAR_WIDTH = 2
    # SheetTracker for a measure stores dataset:column and experiment:row info
    def __init__(self, wb: Workbook, measure: str):
        self.sheet = wb.create_sheet(measure)
        # First row/column is for labels
        self.dataset_column_map = {'header': 1}
        self.exp_row_map = {'header': 1}
        self.exp_label_size = 10

    def add_entry(self, exp, dataset, value):
        if exp not in self.exp_row_map:
            self.exp_row_map[exp] = len(self.exp_row_map) + 1
            self.sheet.cell(self.exp_row_map[exp], 1, exp)
            self.exp_label_size = max(self.exp_label_size, len(dataset))
            self.sheet.column_dimensions['A'].width = self.exp_label_size * self.CHAR_WIDTH
        if dataset not in self.dataset_column_map:
            new_column_number = len(self.dataset_column_map) + 1
            self.dataset_column_map[dataset] = new_column_number
            self.sheet.cell(1, new_column_number, dataset)
            self.sheet.column_dimensions[get_column_letter(new_column_number)].width = 5 * self.CHAR_WIDTH

        self.sheet.cell(self.exp_row_map[exp], self.dataset_column_map[dataset], value)

    def finalize(self):
        # Add color grade to cells based on largest to smallest column-wise
        color_scale = ColorScaleRule(
            start_type='min',
            start_color=Color(rgb='FFFF00'),
            end_type='max',
            end_color=Color(rgb='00FF00')
        )
        for column_no in range(2, len(self.dataset_column_map)+1):
            last_row = len(self.exp_row_map)
            column_letter = get_column_letter(column_no)
            data_range = self.sheet[column_letter+'2':column_letter+str(last_row)]
            vals = [cell.value if cell.value is not None else 0 for row in data_range for cell in row]
            min_value = min(vals)
            max_value = max(vals)

            self.sheet.conditional_formatting.add(
                column_letter + '2:' + column_letter + str(last_row), 
                color_scale
            )

        
def process_summary_data(wb: Workbook, measure_sheet_map: dict, data: dict):
    for measure in data['scores']:
        if measure not in measure_sheet_map:
            measure_sheet_map[measure] = SheetTracker(wb, measure)
        score, variance = data['scores'][measure]
        experiment = data['experiment']
        dataset = data['dataset']
        # measure_sheet_map[measure].add_entry(experiment, dataset, f'{score:.2f}\u00B1{variance:.2f}')
        measure_sheet_map[measure].add_entry(experiment, dataset, float(f'{score:.3f}')*100)


def add_template_data(measure_sheet_map: dict, template_path: str):
    df = pd.read_csv(template_path)
    datasets = [col for col in df if col not in ['measure', 'experiment']]
    for _, row in df.iterrows():
        sheet_tracker = measure_sheet_map[row['measure']]
        experiment = row['experiment']
        for dataset in datasets:
            sheet_tracker.add_entry(experiment, dataset, row[dataset])


if __name__ == '__main__':
    wb = Workbook()
    # Delete default sheet
    del wb['Sheet']
    measure_sheet_map = dict()

    with open('results/summary.json') as ff:
        data = json.load(ff)
        for entry in data:
            process_summary_data(wb, measure_sheet_map, entry)

    add_template_data(measure_sheet_map, 'smd_template.csv')

    for measure, sheet_tracker in measure_sheet_map.items():
        sheet_tracker.finalize()

    output_file = 'results/summary.xlsx'
    logging.info(f'Writing result to {output_file}')
    wb.save(output_file)

