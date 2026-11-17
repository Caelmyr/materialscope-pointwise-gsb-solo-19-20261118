import os
from waitress import serve
from spectra import create_app

app = create_app(os.environ.get('MATERIALSCOPE_DATA_DIR', 'data'))
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', '18922')), threads=4)
