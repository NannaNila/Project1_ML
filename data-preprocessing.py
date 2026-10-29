"""Corrected Sergi preprocessing, fitted on training rows only.

2015 BRFSS code definitions (documentation, no external observations):
https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
Code rules are explicit per field, never inferred from observed maxima.
Only NumPy and the standard library are required at runtime.
"""
import numpy as np

# Compact groups of documented unknown/refused codes. Real responses such as
# EMPLOY1=7 (retired), INCOME2=7, DIABAGE2=77/88 stay intact.
MISSING_GROUPS = {
    (7, 9): """
        LANDLINE GENHLTH HLTHPLN1 PERSDOC2 MEDCOST CHECKUP1 BPHIGH4 BPMEDS BLOODCHO CHOLCHK
        TOLDHI2 CVDSTRK3 ASTHMA3 ASTHNOW CHCSCNCR CHCOCNCR CHCCOPD1 HAVARTH3 ADDEPEV2 CHCKIDNY
        DIABETE3 RENTHOM1 NUMHHOL2 NUMPHON2 CPDEMO1 VETERAN3 INTERNET PREGNANT QLACTLM2
        USEEQUIP BLIND DECIDE DIFFWALK DIFFDRES DIFFALON SMOKE100 SMOKDAY2 STOPSMK2 USENOW3
        EXERANY2 LMTJOIN3 ARTHDIS2 ARTHSOCL SEATBELT FLUSHOT6 PNEUVAC3 HIVTST6 PDIABTST
        PREDIAB1 EYEEXAM DIABEYE DIABEDU CAREGIV1 CRGVLNG1 CRGVHRS1 CRGVPERS CRGVHOUS CRGVMST2
        CRGVEXPT VIPRFVS2 VIEYEXM2 VIINSUR2 CIMEMLOS CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS
        WTCHSALT DRADVISE ASYMPTOM ASTHMED3 ASINHALR HAREHAB1 STREHAB1 CVDASPRN ASPUNSAF
        RLIVPAIN RDUCHART RDUCSTRK ARTTODAY ARTHWGT ARTHEXER ARTHEDU TETANUS HPVADVC2 SHINGLE2
        HADMAM HOWLONG HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM BLDSTOOL
        LSTBLDS3 HADSIGM3 HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME
        PCPSARS1 PCDMDECN SCNTMNY1 SCNTMEL1 SCNTPAID SCNTLPAD SXORIENT TRNSGNDR RCSRLTN2
        CASTHDX2 CASTHNO2 EMTSUPRT LSATISFY MISTMNT ADANXEV DRNKANY5
    """,
    (77, 99): """
        HHADULT PHYSHLTH MENTHLTH POORHLTH INCOME2 LASTSMK2 AVEDRNK2 DRNK3GE5 MAXDRNKS EXRACT11
        EXRACT21 JOINPAIN IMFVPLAC WHRTST10 DOCTDIAB CHKHEMO3 FEETCHK CRGVREL1 CRGVPRB1
        VINOCRE2 HPVADSHT ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE _CRACE1
        _CPRACE _PRACE1 _MRACE1
    """,
    (98, 99): "DIABAGE2 ASTHMAGE ASRCHKUP",
    (9,): """
        MARITAL EDUCA EMPLOY1 INSULIN PCPSADE1 RCSGENDR _CHISPNC _RFHLTH _HCVU651 _RFHYPE5
        _CHOLCHK _RFCHOL _LTASTH1 _CASTHM1 _ASTHMS1 _HISPANC _RACE _RACEG21 _RACEGR3 _RFBMI5
        _CHLDCNT _EDUCAG _INCOMG _SMOKER3 _RFSMOK3 _RFBING5 _RFDRHV5 _FRTLT1 _VEGLT1 _TOTINDA
        _PACAT1 _PAINDX1 _PA150R2 _PA300R2 _PA30021 _PASTRNG _PAREC1 _PASTAE1 _LMTACT1 _LMTWRK1
        _LMTSCL1 _RFSEAT2 _RFSEAT3 _FLSHOT6 _PNEUMO2 _AIDTST3
    """,
    (99,): "CHILDREN",
    (7777, 9999): "WEIGHT2 HEIGHT3",
    (777, 999): """
        ALCDAY5 FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1 EXEROFT1 EXERHMM1 EXEROFT2
        EXERHMM2 STRENGTH BLDSUGAR FEETCHK2 LONGWTCH ASACTLIM
    """,
    (777777, 999999): "FLSHTMY2 HIVTSTD3",
    (7,): "VIDFCLT2 VIREDIF3 VICTRCT4 VIGLUMA2 VIMACDG2 ASATTACK ASNOSLEP",
    (98,): "ASERVIST ASDRVIST",
    (97, 99): "SCNTWRK1 SCNTLWK1",
    (14,): "_AGEG5YR",
    (3,): "_AGE65YR",
    (99999,): "WTKG3",
    (900,): "DROCDY3_",
    (99900,): "_DRNKWEK MAXVO2_ FC60_",
    (99000,): "PAFREQ1_ PAFREQ2_ STRFREQ_",
}
MISSING = {name: codes for codes, names in MISSING_GROUPS.items() for name in names.split()}
# Never having heard of A1C does not specify a measured number of tests.
MISSING['CHKHEMO3'] = (77, 98, 99)

# Documented none/zero codes, limited to quantitative counts.
ZERO_GROUPS = {
    88: "PHYSHLTH MENTHLTH POORHLTH CHILDREN DRNK3GE5 DOCTDIAB CHKHEMO3 FEETCHK ASERVIST ASDRVIST ASRCHKUP ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE",
    888: "STRENGTH BLDSUGAR FEETCHK2 ASACTLIM",
}
ZERO = {name: code for code, names in ZERO_GROUPS.items() for name in names.split()}

# Numeric measurements/counts identified from codebook ranges, not cardinality.
NUMERIC = set("""NUMADULT NUMMEN NUMWOMEN HHADULT PHYSHLTH MENTHLTH POORHLTH DIABAGE2 NUMHHOL2 NUMPHON2 CHILDREN WEIGHT2 HEIGHT3 ALCDAY5 AVEDRNK2 DRNK3GE5 MAXDRNKS FRUITJU1 FRUIT1 FVBEANS FVGREEN FVORANG VEGETAB1 EXEROFT1 EXERHMM1 EXEROFT2 EXERHMM2 STRENGTH JOINPAIN FLSHTMY2 HIVTSTD3 BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK LONGWTCH ASTHMAGE ASERVIST ASDRVIST ASRCHKUP ASACTLIM HPVADSHT PCDMDECN SCNTWRK1 SCNTLWK1 ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE QSTLANG _AGE80 HTIN4 HTM4 WTKG3 _BMI5 DROCDY3_ _DRNKWEK FTJUDA1_ FRUTDA1_ BEANDAY_ GRENDAY_ ORNGDAY_ VEGEDA1_ _MISFRTN _MISVEGN _FRUTSUM _VEGESUM METVL11_ METVL21_ MAXVO2_ FC60_ PADUR1_ PADUR2_ PAFREQ1_ PAFREQ2_ _MINAC11 _MINAC21 STRFREQ_ PAMIN11_ PAMIN21_ PA1MIN_ PAVIG11_ PAVIG21_ PA1VIGM_""".split())

NUMERIC.difference_update({'QSTLANG', 'PCDMDECN'})  # Language / multiple-response codes.

# Administrative/date/sampling fields. _STATE is geography, kept as categorical.
ADMIN = set("""FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU _STSTR
_STRWT _RAWRAKE _WT2RAKE _CLLCPWT _LLCPWT _DUALCOR FLSHTMY2 HIVTSTD3""".split())
# Follow-up conditional on a heart attack; exclude this target-conditioned field.
ADMIN.add("HAREHAB1")
# Prefer existing uniformly measured calculated columns to mixed-unit raw codes.
EQUIVALENTS = {
    'WEIGHT2': 'WTKG3', 'HEIGHT3': 'HTM4', 'ALCDAY5': 'DROCDY3_',
    'FRUITJU1': 'FTJUDA1_', 'FRUIT1': 'FRUTDA1_', 'FVBEANS': 'BEANDAY_',
    'FVGREEN': 'GRENDAY_', 'FVORANG': 'ORNGDAY_', 'VEGETAB1': 'VEGEDA1_',
    'EXEROFT1': 'PAFREQ1_', 'EXEROFT2': 'PAFREQ2_',
    'EXERHMM1': 'PADUR1_', 'EXERHMM2': 'PADUR2_', 'STRENGTH': 'STRFREQ_',
}


def clean_data(x, names):
    """Apply documented field-specific rules without modifying the input."""
    clean = np.where(np.isfinite(x), x, np.nan).copy()
    for j, name in enumerate(names):
        if name in MISSING:
            clean[np.isin(clean[:, j], MISSING[name]), j] = np.nan
        if name in ZERO:
            clean[clean[:, j] == ZERO[name], j] = 0
    return clean


def fit_preprocessing(x, names, nonlinear=False, max_missing=0.8):
    """Fit sparse-column filtering, encoding, medians and scaling on x only."""
    names = np.asarray(names, dtype=str)
    if x.ndim != 2 or len(names) != x.shape[1] or not len(x):
        raise ValueError('Expected nonempty rows and a name for every raw column')
    if not np.isfinite(max_missing) or not 0 <= max_missing < 1:
        raise ValueError('max_missing must be between zero (inclusive) and one (exclusive)')
    clean = clean_data(x, names)
    dropped = ADMIN | {n for n, equivalent in EQUIVALENTS.items() if equivalent in names}
    eligible = (~np.isin(names, list(dropped))) & (np.isnan(clean).mean(axis=0) <= max_missing)
    # Keep missingness variation even when there is only one observed value.
    kept = np.array([j for j in np.flatnonzero(eligible)
                     if len(np.unique(clean[np.isfinite(clean[:, j]), j])) > 1
                     or np.isnan(clean[:, j]).any()], dtype=int)
    numeric = np.array([j for j in kept if names[j] in NUMERIC], dtype=int)
    categories = np.array([j for j in kept if names[j] not in NUMERIC], dtype=int)
    levels, references = [], []
    for j in categories:
        values, counts = np.unique(clean[np.isfinite(clean[:, j]), j], return_counts=True)
        references.append(values[np.argmax(counts)])
        levels.append(values[values != references[-1]])
    medians = np.nanmedian(clean[:, numeric], axis=0)
    filled = np.where(np.isfinite(clean[:, numeric]), clean[:, numeric], medians)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    scales[scales <= 1e-10] = 1
    missing_columns = numeric[np.isnan(clean[:, numeric]).any(axis=0)]
    state = dict(feature_names=names, kept=kept, numeric=numeric,
                 categories=categories, references=np.array(references),
                 levels=np.concatenate(levels) if levels else np.array([]),
                 level_counts=np.array([len(v) for v in levels], dtype=int),
                 missing_columns=missing_columns, medians=medians,
                 means=means, scales=scales)
    if nonlinear:
        # Three predeclared terms from measurements, never arbitrary category codes.
        positions = [int(np.flatnonzero(names[numeric] == name)[0])
                     for name in ('_AGE80', '_BMI5')]
        state['nonlinear_positions'] = np.array(positions, dtype=int)
        terms = nonlinear_terms((filled - means) / scales, state)
        state['nonlinear_means'] = terms.mean(axis=0)
        state['nonlinear_scales'] = np.maximum(terms.std(axis=0), 1e-10)
    tx = transform(x, state)
    # Remove cross-variable dependencies too: the supplied least_squares uses a
    # normal-equation solve, and unregularized logistic/SGD use whitening.
    residual = tx.T @ tx / len(tx)
    independent = []
    for _ in range(tx.shape[1]):
        j = 0 if not independent else int(np.argmax(np.diag(residual)))
        if residual[j, j] <= 1e-8:
            break
        independent.append(j)
        direction = residual[:, j].copy() / np.sqrt(residual[j, j])
        residual -= np.outer(direction, direction)
        residual[j, :] = residual[:, j] = 0
    state['independent_columns'] = np.array(independent, dtype=int)
    return state


def nonlinear_terms(standardized_numeric, state):
    age, bmi = standardized_numeric[:, state['nonlinear_positions']].T
    return np.column_stack((age**2, bmi**2, age * bmi))


def transform(x, state):
    """Apply a saved state. Unknown categories use the reference encoding.

    Category levels seen only outside training never alter the feature layout.
    A missing category has its own indicator. Numeric missingness is retained.
    """
    if x.ndim != 2 or x.shape[1] != len(state['feature_names']):
        raise ValueError('Raw feature layout does not match fitted preprocessing')
    clean = clean_data(x, state['feature_names'])
    columns = clean[:, state['numeric']]
    filled = np.where(np.isfinite(columns), columns, state['medians'])
    blocks = [np.ones((len(x), 1)), (filled - state['means']) / state['scales'],
              np.isnan(clean[:, state['missing_columns']])]
    offset = 0
    for j, count in zip(state['categories'], state['level_counts']):
        values = state['levels'][offset:offset + count]
        blocks.extend([clean[:, j, None] == values, np.isnan(clean[:, j, None])])
        offset += count
    if 'nonlinear_positions' in state:
        terms = nonlinear_terms((filled - state['means']) / state['scales'], state)
        blocks.append((terms - state['nonlinear_means']) / state['nonlinear_scales'])
    tx = np.column_stack(blocks)
    return tx[:, state['independent_columns']] if 'independent_columns' in state else tx
