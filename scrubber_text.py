import re
import functools

from oai.db import nz
df = nz.to_df('select * from spider..note_fact limit 10')

def note_scrubber(df, note_col, scrub_cols):

    def replace_by(row, scrub_cols):
        replace_mapper = {re.escape(str(row[x])): '' for x in scrub_cols}
        pattern = re.compile("|".join(replace_mapper.keys()))
        replaced_text = pattern.sub(lambda m: replace_mapper[re.escape(m.group(0))], row[note_col])
        return replaced_text

    replace_by_scrub = functools.partial(replace_by, scrub_cols=columns_to_scrub_with)
    return df.apply(replace_by_scrub, axis=1)

scrub_cols = ['patient_dim', 'patient_ptr']
note_col = 'note_text'
scrubbed_text = note_scrubber(df, note_col=note_col, scrub_cols=scrub_cols)