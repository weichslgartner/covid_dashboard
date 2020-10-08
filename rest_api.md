country : str: [afghanistan, albania, algeria, andorra, angola, antigua and barbuda, argentina, armenia, australia, austria, azerbaijan, bahamas, bahrain, bangladesh, barbados, belarus, belgium, belize, benin, bhutan, bolivia, bosnia and herzegovina, botswana, brazil, brunei, bulgaria, burkina faso, burma, burundi, cabo verde, cambodia, cameroon, canada, central african republic, chad, chile, china, colombia, comoros, congo (brazzaville), congo (kinshasa), costa rica, cote d'ivoire, croatia, cuba, cyprus, czechia, denmark, diamond princess, djibouti, dominica, dominican republic, ecuador, egypt, el salvador, equatorial guinea, eritrea, estonia, eswatini, ethiopia, fiji, finland, france, gabon, gambia, georgia, germany, ghana, greece, grenada, guatemala, guinea, guinea-bissau, guyana, haiti, holy see, honduras, hungary, iceland, india, indonesia, iran, iraq, ireland, israel, italy, jamaica, japan, jordan, kazakhstan, kenya, korea, south, kosovo, kuwait, kyrgyzstan, laos, latvia, lebanon, lesotho, liberia, libya, liechtenstein, lithuania, luxembourg, ms zaandam, madagascar, malawi, malaysia, maldives, mali, malta, mauritania, mauritius, mexico, moldova, monaco, mongolia, montenegro, morocco, mozambique, namibia, nepal, netherlands, new zealand, nicaragua, niger, nigeria, north macedonia, norway, oman, pakistan, panama, papua new guinea, paraguay, peru, philippines, poland, portugal, qatar, romania, russia, rwanda, saint kitts and nevis, saint lucia, saint vincent and the grenadines, san marino, sao tome and principe, saudi arabia, senegal, serbia, seychelles, sierra leone, singapore, slovakia, slovenia, somalia, south africa, south sudan, spain, sri lanka, sudan, suriname, sweden, switzerland, syria, taiwan*, tajikistan, tanzania, thailand, timor-leste, togo, trinidad and tobago, tunisia, turkey, us, uganda, ukraine, united arab emirates, united kingdom, uruguay, uzbekistan, venezuela, vietnam, west bank and gaza, western sahara, yemen, zambia, zimbabwe]
countries to display

per_capita : bool: ['true', 'false]
cases per million or overall

window_size: int
window size of rolling average

plot_raw: bool
plot the raw values

plot_average: bool
plot values averaged with defined window size

plot_trend: bool
plot trendline (linear interpolation of last window size)

average: str ['mean', 'median']
function used for calucalting average over window 

scale: str ['linear', 'log]
scale used for y axis

data: str ['confirmed', 'deaths', recovered]
what data to plot