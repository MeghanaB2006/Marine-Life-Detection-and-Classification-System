# marine_database.py

MARINE_SPECIES_DB = {

"bangus": {
"edible": True,
"common_name": "Milkfish",
"medicinal_value": "Rich in Omega-3 fatty acids, improves heart health",
"min_mature_size_cm": 30
},

"catfish": {
"edible": True,
"common_name": "Catfish",
"medicinal_value": "High protein, supports muscle and bone strength",
"min_mature_size_cm": 25
},

"gold fish": {
"edible": False,
"common_name": "Goldfish",
"medicinal_value": "Ornamental fish – not for consumption",
"min_mature_size_cm": 6
},

"grass carp": {
"edible": True,
"common_name": "Grass Carp",
"medicinal_value": "Low fat fish useful for digestion and weight control",
"min_mature_size_cm": 40
},

"indian carp": {
"edible": True,
"common_name": "Indian Major Carp",
"medicinal_value": "Rich in DHA and Omega-3, improves brain function",
"min_mature_size_cm": 35
},

"pangasius": {
"edible": True,
"common_name": "Basa Fish",
"medicinal_value": "Rich in vitamins and minerals, boosts immunity",
"min_mature_size_cm": 30
},

"perch": {
"edible": True,
"common_name": "Perch",
"medicinal_value": "High calcium content, good for bone strength",
"min_mature_size_cm": 20
},

"silver carp": {
"edible": True,
"common_name": "Silver Carp",
"medicinal_value": "Boosts immunity and antioxidant levels",
"min_mature_size_cm": 35
},

"snakehead": {
"edible": True,
"common_name": "Snakehead Fish",
"medicinal_value": "Known for wound healing and tissue repair",
"min_mature_size_cm": 30
},

"tilapia": {
"edible": True,
"common_name": "Tilapia",
"medicinal_value": "Supports heart health and muscle growth",
"min_mature_size_cm": 25
},

"goby": {
"edible": True,
"common_name": "Goby Fish",
"medicinal_value": "High protein and beneficial amino acids",
"min_mature_size_cm": 10
},

"mullet": {
"edible": True,
"common_name": "Mullet",
"medicinal_value": "Rich in Omega-3 fatty acids for cardiovascular health",
"min_mature_size_cm": 30
},

"tarpon": {
"edible": True,
"common_name": "Tarpon",
"medicinal_value": "High protein fish beneficial for muscle growth",
"min_mature_size_cm": 40
},

"glass perchlet": {
"edible": True,
"common_name": "Glass Perchlet",
"medicinal_value": "Contains essential minerals and trace nutrients",
"min_mature_size_cm": 12
},

"knife fish": {
"edible": True,
"common_name": "Knifefish",
"medicinal_value": "Rich in lean protein and vitamins",
"min_mature_size_cm": 25
},

"climbing perch": {
"edible": True,
"common_name": "Climbing Perch",
"medicinal_value": "Traditional medicinal fish for recovery diets",
"min_mature_size_cm": 15
},

"mosquito fish": {
"edible": False,
"common_name": "Mosquito Fish",
"medicinal_value": "Used for mosquito population control",
"min_mature_size_cm": 4
},

"mudfish": {
"edible": True,
"common_name": "Mudfish",
"medicinal_value": "High nutrient fish used in traditional diets",
"min_mature_size_cm": 20
},

"silver barb": {
"edible": True,
"common_name": "Silver Barb",
"medicinal_value": "Provides essential proteins and micronutrients",
"min_mature_size_cm": 18
},

"silver perch": {
"edible": True,
"common_name": "Silver Perch",
"medicinal_value": "Rich in protein and low fat",
"min_mature_size_cm": 30
},

"freshwater eel": {
"edible": True,
"common_name": "Freshwater Eel",
"medicinal_value": "Rich in vitamins A and D for immunity",
"min_mature_size_cm": 45
},

"big head carp": {
"edible": True,
"common_name": "Bighead Carp",
"medicinal_value": "Good protein source with Omega-3 fatty acids",
"min_mature_size_cm": 50
},

"green spotted puffer": {
"edible": False,
"common_name": "Green Spotted Puffer",
"medicinal_value": "Contains toxins used in biomedical research",
"min_mature_size_cm": 15
},

"janitor fish": {
"edible": False,
"common_name": "Pleco / Janitor Fish",
"medicinal_value": "Not typically consumed, helps clean algae",
"min_mature_size_cm": 30
},

"black spotted barb": {
"edible": True,
"common_name": "Black Spotted Barb",
"medicinal_value": "Source of essential amino acids",
"min_mature_size_cm": 15
},

"gourami": {
"edible": True,
"common_name": "Gourami",
"medicinal_value": "High nutritional value and protein content",
"min_mature_size_cm": 25
},

"fourfinger threadfin": {
"edible": True,
"common_name": "Fourfinger Threadfin",
"medicinal_value": "Beneficial for heart health",
"min_mature_size_cm": 35
},

"indo-pacific tarpon": {
"edible": True,
"common_name": "Indo-Pacific Tarpon",
"medicinal_value": "High protein and beneficial fatty acids",
"min_mature_size_cm": 35
},

"long snouted pipefish": {
"edible": False,
"common_name": "Pipefish",
"medicinal_value": "Studied in marine biology research",
"min_mature_size_cm": 12
},

"tenpounder": {
"edible": True,
"common_name": "Tenpounder Fish",
"medicinal_value": "Provides minerals beneficial for bone health",
"min_mature_size_cm": 30
},

"jaguar gapote": {
"edible": True,
"common_name": "Jaguar Cichlid",
"medicinal_value": "Protein rich fish used in traditional diets",
"min_mature_size_cm": 35
}

}

# ---------------- FISH ANALYSIS ----------------

def analyze_fish(species_name, estimated_size_cm=None):

    species = MARINE_SPECIES_DB.get(species_name.lower())

    if not species:
        return {
            "growth_stage": "Unknown",
            "medicinal_status": "Unknown",
            "medicinal_value": "Not available"
        }

    if estimated_size_cm:

        if estimated_size_cm < species["min_mature_size_cm"] * 0.5:
            growth_stage = "Juvenile"

        elif estimated_size_cm < species["min_mature_size_cm"]:
            growth_stage = "Growing"

        else:
            growth_stage = "Mature"

    else:
        growth_stage = "Unknown"

    if not species["edible"]:
        medicinal_status = "Not suitable for consumption"

    elif growth_stage == "Mature":
        medicinal_status = "Suitable for medicinal and dietary use"

    else:
        medicinal_status = "Not fully mature for medicinal use"

    return {
        "growth_stage": growth_stage,
        "medicinal_status": medicinal_status,
        "medicinal_value": species["medicinal_value"]
    }


# ---------------- POND HEALTH ANALYSIS ----------------

def analyze_pond_health(species_count_dict):

    total_fish = sum(species_count_dict.values())
    species_variety = len(species_count_dict)

    if species_variety >= 5:
        diversity = "High"

    elif species_variety >= 3:
        diversity = "Moderate"

    else:
        diversity = "Low"

    if total_fish > 30:
        density = "Overcrowded"

    elif total_fish > 12:
        density = "Balanced"

    else:
        density = "Low Population"

    if diversity == "High" and density == "Balanced":
        overall = "Healthy Pond"

    elif density == "Overcrowded":
        overall = "Risk of Low Oxygen – Reduce Fish Density"

    else:
        overall = "Moderate – Monitor Water Quality"

    return {
        "total_fish_detected": total_fish,
        "diversity_level": diversity,
        "density_status": density,
        "pond_health_rating": overall,
        "overall_health_score": f"{85 if overall=='Healthy Pond' else 60}/100"
    }