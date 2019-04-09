from datetime import datetime
from oai import logger
log = logger.global_logger(__name__)
import argparse
from pathlib import Path
from oai.db import mssql
from google.cloud import storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
if __name__ == "__main__":
    #log.info('Starting ROPs Data Sourcing...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_false')
    parser.add_argument('--save-path')
    #parser.add_argument('--push-path')
    args = parser.parse_args()
    print(f'Arguments: {args}')
    if args.smoke_test:
        query = 'SELECT TOP 100 * FROM team_torchlight.npsp.tbl_weekly_manestream_history'
    else:
        query = 'SELECT TOP 1000 * FROM team_torchlight.npsp.tbl_weekly_manestream_history'
    log.info(query)
    df = mssql.to_df(query)
    del df['mpi']
    del df['first_name']
    del df['last_name']
    log.info(f'Saving to {args.save_path}...')
    df.to_pickle(args.save_path)
    from subprocess import check_call
    check_call(['gzip', args.save_path])
    gzip_path = f'{args.save_path}.gz'
    storage_client = storage.Client()
    today = str(datetime.today().strftime('%Y%m%d'))
    bucket_name = 'rops'
    storage_blob = f'data/receipts/predict/source/{today}/df.pkl.gz'
    upload_blob(bucket_name='rops', source_file_name=gzip_path, destination_blob_name=storage_blob)
