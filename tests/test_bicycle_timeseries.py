#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.bicycle_timeseries import bicycle_timeseries, main


class TestBicycleTimeseries(unittest.TestCase):

    def setUp(self):
        self.df = bicycle_timeseries()

    def test_shape(self):
        self.assertEqual(
            self.df.shape,
            (37128, 20),
            msg="bicycle_timeseries() returned a DataFrame of shape %r, "
            "expected (37128, 20) - one row per hourly timestamp and one "
            "column per counting station." % (self.df.shape,),
        )

    def test_columns(self):
        cols = ['Auroransilta', 'Eteläesplanadi', 'Huopalahti (asema)',
                'Kaisaniemi/Eläintarhanlahti', 'Kaivokatu', 'Kulosaaren silta et.',
                'Kulosaaren silta po. ', 'Kuusisaarentie', 'Käpylä, Pohjoisbaana',
                'Lauttasaaren silta eteläpuoli', 'Merikannontie',
                'Munkkiniemen silta eteläpuoli', 'Munkkiniemi silta pohjoispuoli',
                'Heperian puisto/Ooppera', 'Pitkäsilta itäpuoli',
                'Pitkäsilta länsipuoli', 'Lauttasaaren silta pohjoispuoli',
                'Ratapihantie', 'Viikintie', 'Baana']
        np.testing.assert_array_equal(
            self.df.columns,
            cols,
            err_msg="bicycle_timeseries() columns do not match the "
            "expected counting-station columns, or are in the wrong order.",
        )

    def test_index(self):
        self.assertIsInstance(
            self.df.index[0],
            pd.Timestamp,
            msg="The DataFrame's index must contain pandas Timestamp "
            "objects, not raw strings or integers. Got %r."
            % (type(self.df.index[0]),),
        )
        self.assertEqual(
            self.df.index[0],
            pd.to_datetime("2014-1-1 00:00"),
            msg="The first index entry should be 2014-01-01 00:00, got %r."
            % (self.df.index[0],),
        )
        self.assertEqual(
            self.df.index[1],
            pd.to_datetime("2014-1-1 01:00"),
            msg="The second index entry should be 2014-01-01 01:00, got %r."
            % (self.df.index[1],),
        )

    def test_calls(self):
        with patch(
            "src.bicycle_timeseries.bicycle_timeseries", wraps=bicycle_timeseries
        ) as pbts, patch(
            "src.bicycle_timeseries.pd.read_csv", wraps=pd.read_csv
        ) as prc, patch(
            "src.bicycle_timeseries.pd.to_datetime", wraps=pd.to_datetime
        ) as pdatetime:
            main()
            pbts.assert_called_once_with()
            prc.assert_called_once()
            pdatetime.assert_called()


if __name__ == '__main__':
    unittest.main()
