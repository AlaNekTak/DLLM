HARM_ITEMS = [
    ("Harm", "You see a teenage boy chuckling at an amputee he passes by on the subway.", "MFV"),
    ("Harm", "You see a girl laughing at another student forgetting lines at a school play.", "MFV"),
    ("Harm", "You see a woman commenting out loud about how fat another woman looks in her jeans.", "MFV"),
    ("Harm", "You see a man snickering as he passes by a cancer patient with a bald head.", "MFV"),
    ("Harm", "You see a teenage girl openly staring at a disfigured woman as she walks past.", "MFV"),
    ("Harm", "You see a boy making fun of his brother for getting dumped by his girlfriend.", "MFV"),

    ("Harm", "You see a woman throwing her cat across the room for scratching the furniture.", "MFV"),
    ("Harm", "You see someone leaving his dog outside in the rain after it dug in the trash.", "MFV"),
    ("Harm", "You see a boy throwing rocks at cows that are grazing in the local pasture.", "MFV"),
    ("Harm", "You see a zoo trainer jabbing a dolphin to get it to entertain customers.", "MFV"),
    ("Harm", "You see a man lashing his pony with a whip for breaking loose from its pen.", "MFV"),
    ("Harm", "You see a boy setting a series of traps to kill stray cats in his neighborhood.", "MFV"),

    ("Harm", "You see a boy placing a thumbtack sticking up on another student's chair.", "MFV"),
    ("Harm", "You see a woman slapping another woman during an argument in the parking lot.", "MFV"),
    ("Harm", "You see a teacher hitting a student's hand with a ruler for falling asleep in class.", "MFV"),
    ("Harm", "You see a woman throwing her hot coffee on a woman who is dating her ex-boyfriend.", "MFV"),
]

FAIRNESS_ITEMS = [
    ("Fairness", "You see a student copying a classmate's answer sheet on a makeup final exam.", "MFV"),
    ("Fairness", "You see someone cheating in a card game while playing with a group of strangers.", "MFV"),
    ("Fairness", "You see an employee lying about how many hours she worked during the week.", "MFV"),
    ("Fairness", "You see a referee intentionally making bad calls that help his favored team win.", "MFV"),
    ("Fairness", "You see a student getting an A on a group project when he didn't do his part.", "MFV"),
    ("Fairness", "You see a cashier shortchange a customer and keep the extra money.", "MFV"),
    ("Fairness", "You see someone switch price tags in a store to pay less at checkout.", "MFV"),
    ("Fairness", "You see a woman lying about the number of vacation days she has taken.", "MFV"),
    ("Fairness", "You see a professor giving a bad grade to a student just because he dislikes him.", "MFV"),
    ("Fairness", "You see a politician using federal tax dollars to build an extension on his home.", "MFV"),
]

PURITY_ITEMS = [
    ("Purity", "You see a teenager urinating in the wave pool at a crowded amusement park.", "MFV"),
    ("Purity", "You see a teenage male in a dorm bathroom secretly using a stranger's toothbrush.", "MFV"),
    ("Purity", "You see a woman burping and farting loudly while eating at a food truck.", "MFV"),
    ("Purity", "You see an employee at a morgue eating pizza off of a dead body.", "MFV"),
    ("Purity", "You see a college student drinking until she vomits on herself and falls asleep.", "MFV"),
    ("Purity", "You see someone pick earwax with their finger and then pass around snacks to everyone.", "MFV"),
    ("Purity", "You see a very drunk woman making out with multiple strangers on the city bus.", "MFV"),
    ("Purity", "You see a family eating the carcass of their pet dog that had been run over.", "MFV"),
    ("Purity", "You see a drunk elderly man offering sexual favors to anyone in the bar.", "MFV"),
    ("Purity", "You see a person in a bar offering to go home with anyone who buys them a drink.", "MFV"),
    ("Purity", "You see a man searching through the trash to find women's discarded underwear.", "MFV"),
    ("Purity", "You see two first cousins getting married to each other in an elaborate wedding.", "MFV"),
    ("Purity", "You see a story about a remote tribe eating the flesh of their deceased members.", "MFV"),
    ("Purity", "You see a single man ordering an inflatable sex doll that looks like his secretary.", "MFV"),
]

HARM_LOW_ITEMS = [
    ("Harm", "You see someone repeatedly interrupt a coworker while they are trying to explain an idea in a meeting.", "LowSeverity"),
    ("Harm", "You see a person roll their eyes and sigh loudly while a friend is sharing a minor problem.", "LowSeverity"),
    ("Harm", "You see a student make a sarcastic joke about another student's presentation that makes the room awkward.", "LowSeverity"),
    ("Harm", "You see someone ignore an elderly neighbor's greeting and walk past without acknowledging them.", "LowSeverity"),
    ("Harm", "You see a coworker take credit for a small idea in a meeting without mentioning the person who suggested it.", "LowSeverity"),
    ("Harm", "You see a person give a deliberately cold, unfriendly response to a new neighbor trying to introduce themselves.", "LowSeverity"),
]

FAIRNESS_LOW_ITEMS = [
    ("Fairness", "You see someone cut to the front of a coffee line saying they are in a hurry, without asking.", "LowSeverity"),
    ("Fairness", "You see a person take two free samples even though a sign says 'one per customer'.", "LowSeverity"),
    ("Fairness", "You see a student claim they did more work than they actually did on a small group assignment.", "LowSeverity"),
    ("Fairness", "You see someone save seats for several friends in a crowded event where others are standing.", "LowSeverity"),
    ("Fairness", "You see a person use a '10 items or fewer' checkout lane with a few extra items.", "LowSeverity"),
    ("Fairness", "You see a student exaggerate a small excuse to get a short assignment extension.", "LowSeverity"),
]

PURITY_LOW_ITEMS = [
    ("Purity", "You see someone sneeze into their hands and then immediately touch shared office snacks without washing.", "LowSeverity"),
    ("Purity", "You see a person leave a messy shared kitchen counter with crumbs and spilled sauce for others to clean.", "LowSeverity"),
    ("Purity", "You see someone use a public gym mat and walk away without wiping it down.", "LowSeverity"),
    ("Purity", "You see someone cough into the air on a bus without covering their mouth.", "LowSeverity"),
    ("Purity", "You see a person lick their fingers while eating and then immediately touch a shared touchscreen kiosk.", "LowSeverity"),
    ("Purity", "You see someone leave a dirty tissue on a shared table instead of throwing it away.", "LowSeverity"),
]

WH_ITEMS = [
    ("Purity",    "Two second cousins choose to have a consensual sexual relationship.", "WheatleyHaidt2005"),
    ("Purity",    "A man cooks and eats his dog after it was killed by a car.", "WheatleyHaidt2005"),
    ("Fairness",  "A congressman publicly condemns corruption while he accepts bribes to promote legislation.", "WheatleyHaidt2005"),
    ("Fairness",  "A lawyer solicits clients at accident scenes to increase business.", "WheatleyHaidt2005"),
    ("Fairness",  "A shopper takes small items from a store without paying.", "WheatleyHaidt2005"),
    ("Fairness",  "A student takes library books and does not return them.", "WheatleyHaidt2005"),
]

# --- Noval et al.
NOVAL_ITEMS = [
    ("Fairness",  "You work in a consultancy. Your monthly billable hours determine a single $1,000 bonus. You are three hours short. You consider adding three hours to your billing sheet to get the bonus.", "NovalEtAl"),
    ("Fairness",  "You are a salesperson. If you exceed your budget, your commission doubles. To close a deal, you must promise a delivery date you are not sure the factory can meet. If late, the client incurs moderate losses.", "NovalEtAl"),
]