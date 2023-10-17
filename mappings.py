def get_setting_word(concept, setting):
	words = {
		'maximum': {
			'plausible': 'coveted',
			'nonsense': 'kazvty',
			'adversarial': 'least'
		},
		'minimum': {
			'plausible': 'baseline',
			'nonsense': 'vlirp',
			'adversarial': 'largest'
		},
		'average': {
			'plausible': 'representative',
			'nonsense': 'erpyiz',
			'adversarial': 'least-frequent'
		},
		'count': {
			'plausible': 'magnitude',
			'nonsense': 'jlorqpe',
			'adversarial': 'subtraction'
		},
		'sum': {
			'plausible': 'accumulation',
			'nonsense': 'tosfke',
			'adversarial': 'frequency'
		},
		'not_in': {
			'plausible': 'absent',
			'nonsense': 'lerfym',
			'adversarial': 'present'
		},
		'more_than_max': {
			'plausible': 'dominate',
			'nonsense': 'bqovr',
			'adversarial': 'yield'
		},
		'second_max': {
			'plausible': 'runner-up',
			'nonsense': 'mlevzgh',
			'adversarial': 'penultimate'
		},
		'above_average': {
			'plausible': 'satisfactory',
			'nonsense': 'rgiuel',
			'adversarial': 'below-average'
		},
		'value_mode': {
			'plausible': 'prevalent',
			'nonsense': 'ifoqas',
			'adversarial': 'least-frequent'
		},
		'salary_more_than': {
			'plausible': 'overpaid',
			'nonsense': 'qroyhst',
			'adversarial': 'underpaid'
		},
		'credit_4': {
			'plausible': 'heavy',
			'nonsense': 'lkefoiy',
			'adversarial': 'lightweight'
		},
		'dept_science': {
			'plausible': 'pure-science',
			'nonsense': 'dhwoisp',
			'adversarial': 'humanities'
		},
		'yellow_card': {
			'plausible': 'aggressive',
			'nonsense': 'giwzle',
			'adversarial': 'meek'
		},
		'city_mv_pa': {
			'plausible': 'tech-towns',
			'nonsense': 'arocfpu',
			'adversarial': 'east-coast'
		},
		'course_prereq_id': {
			'plausible': 'requirement-id',
			'nonsense': 'iregtas',
			'adversarial': 'quotient'
		},
		'lname_fname': {
			'plausible': 'alias',
			'nonsense': 'huwqsox',
			'adversarial': 'middlename'
		},
		'end_start_date': {
			'plausible': 'tenure',
			'nonsense': 'fyxplis',
			'adversarial': 'occurrences'
		},
		'len_less_than': {
			'plausible': ''
		},
		'outlier_range': {
			'plausible': ''
		},
		'below_average': {
			'plausible': ''
		},
		'hire_date': {
			'plausible': ''
		},
		'not_intern': {
			'plausible': ''
		},
		'dock_count': {
			'plausible': ''
		}
	}
	return words[concept][setting]

def get_instruction(concept, setting):
	instructions = {
		'maximum': 'concept_word refers to the maximum value.',
		'minimum': 'concept_word refers to the minimum value.',
		'average': 'concept_word referes to the average value.',
		'count': 'concept_word refers to the count or quantity of.',
		'sum': 'concept_word refers to the sum.',
		'not_in': 'concept_word refers to not doing or not having or not be included in something.',
		'more_than_max': 'concept_word refers to all the column values that are greater than the conditionally maximum column value.',
		'second_max': 'concept_word refers to the second highest value.',
		'above_average': 'concept_word refers to all the column values that are greater than the conditionally average column value.',
		'value_mode': 'concept_word refers to the value being the most frequent or equal to the mode value.',
		'salary_more_than': 'concept_word refers to those with salary more than 30000.',
		'credit_4': 'concept_word refers to number of credits equal to 4.',
		'dept_science': 'concept_word refers to the Physics or Biology subjects.',
		'yellow_card': 'concept_word refers to player having been given a yellow card.',
		'city_mv_pa': 'concept_word refers to the city being either Mountain View or Palo Alto.',
		'course_prereq_id': 'concept_word refers to the product of the course and prerequisite ids.',
		'lname_fname': 'concept_word refers to the concatenation of lastname and firstname.',
		'end_start_date': 'concept_word refers to the difference between end date and start date.',
		'len_less_than': 'the system length constraints are that the length of the value should be less than 8.',
		'outlier_range': 'the first order outlier range consists of values greater than the difference of the mean and standard deviation values.',
		'below_average': 'the community-mandated spectrum refers to values that are less than the average value.',
		'hire_date': 'the months of union labour strike were July 1987 and August 1987.',
		'not_intern': 'board-certified and licensed refers to the positions that are not "Staff Internist".',
		'dock_count': 'biking association compliant stations are those that have a dock count of at least 19.'
	}
	temp_instr = instructions[concept]
	setting_word = get_setting_word(concept, setting)
	final_instr = temp_instr.replace('concept_word', setting_word)
	return final_instr