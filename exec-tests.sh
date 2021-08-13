#!/bin/bash
python -m unittest test_baseconv.py
python -m unittest test_recurrentconv.py
python -m unittest test_convlstm.py
python -m unittest test_convgru.py
python -m unittest test_denseconv.py