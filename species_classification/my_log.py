import csv
import os

class Writer:
   def __init__(self, file_name, header):

      if not (os.path.isfile(file_name)):
         self.output = csv.writer(open(file_name, 'a'))
         self.write_data(header)
      else:
         self.output = csv.writer(open(file_name, 'a'))

   def write_data(self, row_data):
      self.output.writerow(row_data)
