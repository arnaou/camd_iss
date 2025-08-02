import pandas as pd

class Dataset:
    def __init__(self, file_path, output_format='dataframe'):
        """
        Initializes the Dataset object by loading group contribution data and UNIFAC parameters.

        :param file_path: Path to the Excel file containing group contribution data.
        :param output_format: Default format to return the data ('dataframe' or 'dict').
        """
        self.file_path = file_path
        self.default_output_format = output_format
        self.gc_data, self.unifac_data = self._load_gc()

    def _load_gc(self):
        """
        Loads group contribution data and UNIFAC parameters from the specified file.

        :return: Tuple containing group contribution data and UNIFAC parameters.
        """
        # Define sheet names and corresponding column names
        sheet_columns = {
            'set': 'set',       # groups: main groups and subgroups
            'noc': 'noc',       # Number of connections
            'Rk': 'Rk',         # volume parameter of group k
            'Qk': 'Qk',         # surface area parameter of group k
            'Mw': 'Mw',         # Molecular weight in g/mol
            'Tm': 'Tm',         # Normal melting point in K
            'Tb': 'Tb',         # Normal boiling point in K
            'Tc': 'Tc',         # Critical temperature in K
            'Pc': 'Pc',         # Critical pressure in bar
            'Vc': 'Vc',         # Critical volume in cm3/mol
            'Gf': 'Gf',         # Gibbs free energy of formation in kJ/mol
            'Hf': 'Hf',         # Enthalpy of formation in kJ/mol
            'Hv': 'Hv',         # Enthalpy of vaporization in kJ/mol
            'Ac': 'Ac',         # Acentric factor in -
            'Vm': 'Vm',         # Molar volume in cm3/mol
            'tfc': 'tfc',       # closed cup flash point in K
            'tfo': 'tfo',       # open cup flash point in K
            'FDX': 'FDX',       # hansen dispersion solubility parameter MPa^1/2
            'FPX': 'FPX',       # hansen polar solubility parameter MPa^1/2
            'FHX': 'FHX',       # hansen hydrogen solubility parameter MPa^1/2
            'LC50': 'LC50'      # Acute toxicity (96h- LC50) to the fathead minnow
        }

        # Read the Excel file
        xls = pd.ExcelFile(self.file_path)

        # Load group contribution data
        gc_df = pd.concat(
            [pd.read_excel(xls, sheet_name=sheet, usecols="C" if sheet == 'set' else "B") for sheet in sheet_columns.keys()],
            axis=1
        )
        gc_df.columns = sheet_columns.values()
        gc_df.set_index('set', inplace=True)
        gc_df.rename_axis(None, inplace=True)

        # Extract Rk and Qk for UNIFAC parameters
        unifac_r = gc_df['Rk']
        unifac_q = gc_df['Qk']

        # Remove Rk and Qk from gc_df
        gc_df.drop(columns=['Rk', 'Qk'], inplace=True)

        # Load interaction parameters (a)
        unifac_a = pd.read_excel(xls, sheet_name='a', index_col=0)
        unifac_a.drop(unifac_a.index[0], inplace=True)  # Remove the first row with main group data
        unifac_a = unifac_a.apply(pd.to_numeric, errors='coerce').astype('float64')

        return gc_df, {'a': unifac_a, 'q': unifac_q, 'r': unifac_r}

    def _convert_to_dict(self, data):
        """
        Converts a DataFrame or Series to a dictionary.

        :param data: DataFrame or Series to convert.
        :return: Dictionary representation of the data.
        """
        if isinstance(data, pd.DataFrame):
            return data.stack().to_dict()
        elif isinstance(data, pd.Series):
            return data.to_dict()
        else:
            raise ValueError("Unsupported data type for conversion to dict.")

    def get_contribs(self, output_format=None):
        """
        Retrieves the group contribution data.

        :param output_format: Format to return the data ('dataframe' or 'dict').
                              If None, uses the default format.
        :return: Group contribution data in the specified format.
        """
        output_format = output_format or self.default_output_format
        if output_format == 'dataframe':
            return self.gc_data
        elif output_format == 'dict':
            return self._convert_to_dict(self.gc_data)
        else:
            raise ValueError("Invalid output_format. Choose either 'dataframe' or 'dict'.")

    def get_unifac(self, output_format=None):
        """
        Retrieves the UNIFAC parameters.

        :param output_format: Format to return the data ('dataframe' or 'dict').
                              If None, uses the default format.
        :return: UNIFAC parameters in the specified format.
        """
        output_format = output_format or self.default_output_format
        if output_format == 'dataframe':
            return self.unifac_data
        elif output_format == 'dict':
            return {key: self._convert_to_dict(value) for key, value in self.unifac_data.items()}
        else:
            raise ValueError("Invalid output_format. Choose either 'dataframe' or 'dict'.")
    
def main():
    # Initialize the Dataset
    dataset = Dataset(file_path="data/data_unifac.xlsx", output_format="dataframe")

    # Access group contribution data as DataFrame
    contribs_df = dataset.get_contribs(output_format="dataframe")
    print("Group Contribution Data (DataFrame):")
    print(contribs_df)

    # Access group contribution data as dict
    contribs_dict = dataset.get_contribs(output_format="dict")
    print("Group Contribution Data (Dict):")
    print(contribs_dict)

    # Access UNIFAC parameters as DataFrame
    unifac_df = dataset.get_unifac(output_format="dataframe")
    print("UNIFAC Parameters (DataFrame):")
    print(unifac_df)

    # Access UNIFAC parameters as dict
    unifac_dict = dataset.get_unifac(output_format="dict")
    print("UNIFAC Parameters (Dict):")
    #print(unifac_dict)

if __name__ == "__main__":
    main()
