SYNAPSE_TABLES = {
    # 'day_one': 'syn16782072',
    # 'seven_day_activity_sleep': 'syn16782059',
    # 'par_q': 'syn16782071',
    # 'daily_check': 'syn16782070',
    # 'activity_sleep': 'syn16782069',
    # 'risk_factor': 'syn16782068',
    # 'cardio_diet': 'syn16782067',
    # 'satisfied': 'syn16782066',
    # 'aph_heart_rate': 'syn16782065',
    # 'six_minute_walk': 'syn16782064',
    # 'six_minute_walk_displacements': 'syn16782058',
    # 'motion_tracker': 'syn16782059',
    'healthkit_data': 'syn16782062',
    # 'healthkit_sleep': 'syn16782061',
    # 'healthkit_workout': 'syn16782060',
    # 'demographics': 'syn16782063'
}

TABLES_EMBEDDED_DATA_COLUMN = {
    'healthkit_data': 'data.csv',
    # 'healthkit_sleep': 'data.csv',
    # 'healthkit_workout': 'data.csv',
    # 'motion_tracker': 'data.csv',
    # 'six_minute_walk': ['pedometer_fitness.walk.items',
    #                     'accel_fitness_walk.json.items',
    #                     'deviceMotion_fitness.walk.items',
    #                     'HKQuantityTypeIdentifierHeartRate_fitness.walk.items',
    #                     'accel_fitness_rest.json.items',
    #                     'deviceMotion_fitness.rest.items',
    #                     'HKQuantityTypeIdentifierHeartRate_fitness.rest.items'],
    # 'six_minute_walk_displacements': 'UnknownFile_1.json.items'
}

DAILY_SOURCES = [
    "com.beurer.connect",
    "com.ihealthlabs.iHealth",
    "com.ihealthlabs.iHealth02",
    "com.mioglobal.go",
    "com.redshiftdev.Fitbit-Health-Sync",
    "com.runtastic.iphone",
    "com.sunnystudio.uptrending",
    "com.withings.wiScaleNG",
    "mortadelanetwork.FitAllSync",
    "mortadelanetwork.Syncbit",
    "net.jaiyo.misfit-sync:",
    "net.jaiyo.wristbandsync-jawboneup",
    "org.medhelp.MyCycles"
]
