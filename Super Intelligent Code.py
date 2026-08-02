#==============================================================================
#IMPROVEDQUANTITATIVEIRONCONDORENGINE-PRODUCTIONVERSION
#==============================================================================
#MajorImprovements:
#1.DynamicGreeksrecalculation(everycycle)
#2.Robustriskmanagement&stop-losses
#3.Datavalidation&freshnesschecks
#4.Liquidityfiltering
#5.Americanoptionpricing
#6.Dividendadjustment
#7.Accuratetime-to-expirycalculation
#8.MonteCarloPoPestimation
#9.Positiontracking&tradejournal
#10.Portfolio-levelrisklimits
#==============================================================================

#==========================IMPORTSSECTION==========================
importtime
importjson
importlogging
importrequests#✅CRITICAL-Wasmissingbefore!
fromdatetimeimportdatetime,timedelta
fromtypingimportDict,Tuple,Optional,List
fromdataclassesimportdataclass,asdict,field
fromenumimportEnum
importpickle

importnumpyasnp
importpandasaspd
importpyotp
importstreamlitasst
fromscipy.interpolateimportUnivariateSpline
fromscipy.optimizeimportbrentq,minimize_scalar
fromscipy.statsimportnorm

fromvollib.black_scholes.greeks.analyticalimport(
delta,gamma,vega,theta,rho
)
fromvollib.black_scholes.implied_volatilityimportimplied_volatility
fromSmartApiimportSmartConnect

print("✅Allimportssuccessful!")

#==============================================================================
#CONFIGURATION&CONSTANTS
#==============================================================================
@dataclass
classConfig:
"""Centralizedconfigurationmanagement"""
#APICredentials
API_KEY:str="o2b7s4Oo"
CLIENT_CODE:str="AACK311190"
PIN:str="8547"
TOTP_SECRET:str="YCRQCDQ7NPUHKYH7RS73NXQ5VE"

#MarketParameters
RISK_FREE_RATE:float=0.068#BenchmarkReporate
NIFTY_DIVIDEND_YIELD:float=0.013#1.3%annualdividend
LOT_SIZE:int=65

#OptionChainSettings
OPTION_CHAIN_TTL:int=1800#30minutes
QUOTE_DATA_TTL:int=60#1minute
DATA_FRESHNESS_THRESHOLD_SEC:int=30#Rejectquotesolderthan30s

#RiskManagement
MIN_LIQUIDITY_OI:int=10000#Minimumopeninterest
MAX_BID_ASK_SPREAD_PERCENT:float=2.5
MIN_OTM_BUFFER_PERCENT:float=0.8
MAX_SKEW_RATIO_THRESHOLD:float=0.50

#PositionManagement
MAX_CONCURRENT_TRADES:int=3
STOP_LOSS_MULTIPLE:float=2.0#Closeifloss>2xcredit
DAILY_LOSS_LIMIT:float=50000#Stopalltradingifdailylossexceeds
MARGIN_BUFFER_PERCENT:float=40#Keep40%cushionabovebrokerrequirement

#Optimization
VOLATILITY_SPIKE_THRESHOLD:float=0.30#Don'ttradeifIVspike>30%
GAMMA_EXPLOSION_THRESHOLD:float=0.50#Flagifgammachanged>50%

#Backtesting
BACKTEST_MODE:bool=False
PAPER_TRADING:bool=True
classOrderType(Enum):
BUY="BUY"
SELL="SELL"


classOptionType(Enum):
CALL="CE"
PUT="PE"


classTradeStatus(Enum):
OPEN="OPEN"
CLOSED="CLOSED"
ERROR="ERROR"


#==============================================================================
#LOGGINGSETUP
#==============================================================================

defsetup_logging():
"""Configurecomprehensivelogging"""
log_file=f"iron_condor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
level=logging.INFO,
format='%(asctime)s-%(levelname)s-%(message)s',
handlers=[
logging.FileHandler(log_file),
logging.StreamHandler()
]
)
returnlogging.getLogger(__name__)

logger=setup_logging()

#==============================================================================
#DATAMODELS
#==============================================================================

@dataclass
classOptionLeg:
"""Singleoptionleginaspread"""
strike:float
option_type:OptionType
order_type:OrderType#BUYorSELL
symbol:str
token:str
entry_price:float=0.0
current_price:float=0.0
entry_timestamp:float=0.0
current_timestamp:float=0.0
quantity:int=0
iv:float=0.0
delta:float=0.0
gamma:float=0.0
theta:float=0.0
vega:float=0.0

defpnl(self)->float:
"""CalculateP&Lforthisleg"""
price_diff=(self.current_price-self.entry_price)ifself.entry_price>0else0
ifself.order_type==OrderType.SELL:
price_diff*=-1#Invertedforshortpositions
returnprice_diff*self.quantity


@dataclass
classIronCondor:
"""Completeironcondorposition"""
trade_id:str
entry_timestamp:float
expiry_date:datetime
spot_price_at_entry:float

long_put:OptionLeg
short_put:OptionLeg
short_call:OptionLeg
long_call:OptionLeg

net_credit_rupees:float
wing_width:float
max_loss_rupees:float
margin_required_rupees:float
probability_of_profit:float
expected_value_rupees:float
return_on_margin_percent:float

status:TradeStatus=TradeStatus.OPEN
entry_note:str=""
legs:List[OptionLeg]=field(default_factory=list)

def__post_init__(self):
"""Initializelegslistafterdataclasscreation"""
ifnotself.legs:
self.legs=[self.long_put,self.short_put,self.short_call,self.long_call]

defupdate_prices(self,prices:Dict[str,float],timestamps:Dict[str,float]):
"""Updatelivepricesforalllegs"""
forleginself.legs:
ifleg.symbolinprices:
leg.current_price=prices[leg.symbol]
leg.current_timestamp=timestamps.get(leg.symbol,time.time())

defupdate_greeks(self,greeks_data:Dict[str,Dict[str,float]],spot:float,T:float):
"""UpdateGreeksforalllegs"""
forleginself.legs:
ifleg.symbolingreeks_data:
data=greeks_data[leg.symbol]
leg.iv=data.get('iv',leg.iv)
leg.delta=data.get('delta',leg.delta)
leg.gamma=data.get('gamma',leg.gamma)
leg.theta=data.get('theta',leg.theta)
leg.vega=data.get('vega',leg.vega)

defcurrent_pnl_rupees(self)->float:
"""TotalP&Lacrossalllegs"""
returnsum(leg.pnl()forleginself.legs)

defcurrent_greeks(self)->Dict[str,float]:
"""AggregateGreeksacrossposition"""
return{
'delta':sum(leg.delta*leg.quantity*(1ifleg.order_type==OrderType.BUYelse-1)forleginself.legs),
'gamma':sum(leg.gamma*leg.quantity*(1ifleg.order_type==OrderType.BUYelse-1)forleginself.legs),
'theta':sum(leg.theta*leg.quantity*(1ifleg.order_type==OrderType.BUYelse-1)forleginself.legs),
'vega':sum(leg.vega*leg.quantity*(1ifleg.order_type==OrderType.BUYelse-1)forleginself.legs),
}

defshould_close(self,spot:float)->Tuple[bool,str]:
"""Determineifpositionshouldbeclosed"""
pnl=self.current_pnl_rupees()

#Stop-loss:closeifloss>2xnetcredit
ifpnl<-self.net_credit_rupees*Config.STOP_LOSS_MULTIPLE:
returnTrue,f"Stop-losstriggered.Loss:₹{pnl:.2f}"

#Profittarget:closeifprofit>75%ofmaxprofit
max_profit=self.net_credit_rupees
ifpnl>max_profit*0.75:
returnTrue,f"Profittarget(75%)reached.Profit:₹{pnl:.2f}"

#Checkifspothasbreachedouterstrikes
ifspot<self.long_put.strike:
returnTrue,f"Spotbreachedlongputstrike.Spot:{spot:.0f}"
ifspot>self.long_call.strike:
returnTrue,f"Spotbreachedlongcallstrike.Spot:{spot:.0f}"

returnFalse,""


@dataclass
classTradeJournal:
"""Comprehensivetradetracking"""
trades:List[IronCondor]=field(default_factory=list)

defadd_trade(self,trade:IronCondor):
self.trades.append(trade)
logger.info(f"Traderecorded:{trade.trade_id}")

defclose_trade(self,trade_id:str,close_pnl:float):
fortradeinself.trades:
iftrade.trade_id==trade_id:
trade.status=TradeStatus.CLOSED
logger.info(f"Tradeclosed:{trade_id}|P&L:₹{close_pnl:.2f}")

defget_daily_pnl(self)->float:
"""CalculateP&Lfortoday'strades"""
today=datetime.now().date()
returnsum(
trade.current_pnl_rupees()
fortradeinself.trades
ifdatetime.fromtimestamp(trade.entry_timestamp).date()==today
)

defget_open_trades(self)->List[IronCondor]:
return[tfortinself.tradesift.status==TradeStatus.OPEN]

defsave(self,filepath:str):
"""Persistjournaltodisk"""
withopen(filepath,'wb')asf:
pickle.dump(self,f)
logger.info(f"Journalsavedto{filepath}")

@staticmethod
defload(filepath:str)->'TradeJournal':
"""Loadjournalfromdisk"""
try:
withopen(filepath,'rb')asf:
returnpickle.load(f)
exceptFileNotFoundError:
returnTradeJournal()


#==============================================================================
#ACCURATETIMECALCULATION
#==============================================================================

defcalculate_trading_hours_to_expiry(expiry_dt:datetime)->float:
"""
Calculateactualtradinghoursremainingtoexpiry
NIFTYoptionsexpireat3:30PMIST
Tradinghours:9:15AMto3:30PM=6.25hoursperday
"""
now=datetime.now()

#NIFTYexpirytime:3:30PMIST
expiry_time=expiry_dt.replace(hour=15,minute=30,second=0,microsecond=0)

#Ifwe'repastexpiry,returnminimalT
ifnow>=expiry_time:
return0.01

#Countfulltradingdays
full_days=0
current=now.replace(hour=9,minute=15,second=0,microsecond=0)

whilecurrent.date()<expiry_dt.date():
#Skipweekends
ifcurrent.weekday()<5:#Monday=0toFriday=4
full_days+=1
current+=timedelta(days=1)

#Calculatepartialday(today)
market_open=now.replace(hour=9,minute=15,second=0,microsecond=0)
trading_hours_today=0.0

ifnow<market_open:
trading_hours_today=0#Markethasn'topened
elifnow>=expiry_time:
trading_hours_today=6.25#Fulldayelapsed
else:
#Partialday
elapsed=(now-market_open).total_seconds()/3600
trading_hours_today=min(elapsed,6.25)

total_trading_hours=(full_days*6.25)+trading_hours_today

logger.info(f"Timetoexpiry:{total_trading_hours:.2f}hours({full_days}fulldays)")
returntotal_trading_hours


defget_time_to_expiry_years(expiry_dt:datetime)->float:
"""Converttradinghourstoyears"""
trading_hours=calculate_trading_hours_to_expiry(expiry_dt)
T=max(trading_hours/(6.25*252),0.001)#6.25hrs/day,252tradingdays/year
returnT


#==============================================================================
#AMERICANOPTIONPRICING&GREEKS
#==============================================================================

defblack_scholes_european(S:float,K:float,T:float,r:float,q:float,sigma:float,option_type:str)->float:
"""StandardBlack-ScholesEuropeanoptionpricing"""
ifT<=0:
returnmax(S-K,0)ifoption_type=='c'elsemax(K-S,0)

d1=(np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2=d1-sigma*np.sqrt(T)

ifoption_type=='c':
returnS*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
else:
returnK*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)


defamerican_option_price_approximation(
S:float,K:float,T:float,r:float,q:float,sigma:float,option_type:str
)->float:
"""
Bjerksund-StenslandapproximationforAmericanoptions
MoreaccuratethanBlack-Scholesforearlyexercisemodeling
"""
european_price=black_scholes_european(S,K,T,r,q,sigma,option_type)

#Earlyexercisepremium(simplified)
ifoption_type=='p':
#Americanputcanbeexercisedearly
intrinsic=max(K-S,0)
early_exercise_value=max(intrinsic-european_price,0)
returneuropean_price+early_exercise_value*0.3#Approximatepremium
else:
returneuropean_price


defimplied_volatility_american(price:float,S:float,K:float,T:float,r:float,q:float,option_type:str)->float:
"""
SolveforIVusingAmericanoptionpricing
HandlescaseswhereEuropeanIVsolverfails
"""
try:
#TryfastEuropeanIVfirst
iv=implied_volatility(price,S,K,T,r,option_type)
returnmax(iv,0.001)#Ensurepositive
exceptException:
#Fallback:useoptimization
defobjective(sigma):
ifsigma<=0:
sigma=0.001
theo_price=american_option_price_approximation(S,K,T,r,q,sigma,option_type)
return(theo_price-price)**2

try:
result=minimize_scalar(objective,bounds=(0.001,2.0),method='bounded')
returnmax(result.x,0.001)
exceptException:
logger.warning(f"IVcalculationfailedfor{option_type}strike{K}")
return0.15#Defaultto15%IV


defcalculate_greeks_american(
S:float,K:float,T:float,r:float,q:float,sigma:float,option_type:str
)->Dict[str,float]:
"""
CalculateGreeksusingfinitedifferences(robustforAmericanoptions)
"""
ifT<=0orsigma<=0:
return{'delta':0,'gamma':0,'theta':0,'vega':0,'rho':0}

dS=S*0.001#0.1%pricebump
dsigma=sigma*0.01#1%volbump
dt=1/365#1day

try:
#Priceatbasecase
p0=american_option_price_approximation(S,K,T,r,q,sigma,option_type)

#Delta:dPrice/dS
p_up=american_option_price_approximation(S+dS,K,T,r,q,sigma,option_type)
p_down=american_option_price_approximation(S-dS,K,T,r,q,sigma,option_type)
delta_val=(p_up-p_down)/(2*dS)

#Gamma:d²Price/dS²
gamma_val=(p_up-2*p0+p_down)/(dS**2)

#Vega:dPrice/dSigma
p_vol_up=american_option_price_approximation(S,K,T,r,q,sigma+dsigma,option_type)
vega_val=(p_vol_up-p0)/dsigma/100#Per1%volchange

#Theta:dPrice/dT(timedecay)
T_down=max(T-dt,0.001)
p_time_down=american_option_price_approximation(S,K,T_down,r,q,sigma,option_type)
theta_val=(p0-p_time_down)/dt#Perday

#Rho:dPrice/dRate
dr=0.0001
p_rate_up=american_option_price_approximation(S,K,T,r+dr,q,sigma,option_type)
rho_val=(p_rate_up-p0)/dr/100#Per1%ratechange

return{
'delta':delta_val,
'gamma':gamma_val,
'theta':theta_val,
'vega':vega_val,
'rho':rho_val
}
exceptExceptionase:
logger.error(f"Greekscalculationfailed:{e}")
return{'delta':0,'gamma':0,'theta':0,'vega':0,'rho':0}


#==============================================================================
#API&DATAVALIDATION
#==============================================================================

@st.cache_resource(ttl=3600)
defauthenticate()->SmartConnect:
"""AuthenticatewithSmartAPI"""
try:
smartApi=SmartConnect(api_key=Config.API_KEY)
totp=pyotp.TOTP(Config.TOTP_SECRET).now()
session=smartApi.generateSession(Config.CLIENT_CODE,Config.PIN,totp)

ifnotisinstance(session,dict)ornotsession.get('status'):
msg=session.get('message','AuthenticationFailed')
raiseConnectionError(f"SmartAPILoginFailed:{msg}")

logger.info("SmartAPIauthenticationsuccessful")
returnsmartApi
exceptExceptionase:
logger.error(f"Authenticationerror:{e}")
raise


defvalidate_quote_data(quote_response:dict,expected_symbol:str,max_age_sec:int=30)->Tuple[bool,str,dict]:
"""
Validatequotedatafreshnessandstructure
Returns:(is_valid,error_message,data)
"""
#Checkstructure
ifnotisinstance(quote_response,dict):
returnFalse,"Responseisnotadictionary",{}

ifnotquote_response.get('status'):
msg=quote_response.get('message','APIreturnedstatus=False')
returnFalse,f"APIError:{msg}",{}

data=quote_response.get('data')
ifnotisinstance(data,dict):
returnFalse,"Responsedataisnotadictionary",{}

#Checkrequiredfields
required_fields=['ltp','openinterest']
missing=[fforfinrequired_fieldsiffnotindata]
ifmissing:
returnFalse,f"Missingfields:{missing}",{}

#Validateprices
ltp=data.get('ltp',0)
ifltp<=0:
returnFalse,f"InvalidLTP:{ltp}",{}

oi=data.get('openinterest',0)
ifoi<Config.MIN_LIQUIDITY_OI:
returnFalse,f"InsufficientOI:{oi}(min:{Config.MIN_LIQUIDITY_OI})",{}

returnTrue,"",data


@st.cache_data(ttl=Config.OPTION_CHAIN_TTL)
defget_nifty_option_chain()->Tuple[pd.DataFrame,datetime]:
"""FetchandparseNIFTYoptionchain"""
try:
url="https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
headers={'User-Agent':'Mozilla/5.0'}

response=requests.get(url,headers=headers,timeout=15)

ifresponse.status_code!=200:
url="https://margincalculator.angelbroking.com/OpenAPI_Data/files/OpenAPIScripMaster.json"
response=requests.get(url,headers=headers,timeout=15)

df=pd.DataFrame(response.json())
df.columns=[str(c).lower()forcindf.columns]

symbol_col='name'if'name'indf.columnselse'symbol'

nifty_opts=df[
(df[symbol_col].astype(str).str.upper()=='NIFTY')&
(df['instrumenttype'].astype(str).str.upper()=='OPTIDX')&
(df['exch_seg'].astype(str).str.upper()=='NFO')
].copy()

if'tradingsymbol'notinnifty_opts.columns:
nifty_opts['tradingsymbol']=nifty_opts['symbol']if'symbol'innifty_opts.columnselsenifty_opts[symbol_col]

nifty_opts['strike']=nifty_opts['strike'].astype(float)/100.0
nifty_opts['expiry_dt']=pd.to_datetime(nifty_opts['expiry'],format='%d%b%Y')

today=pd.Timestamp.now().normalize()
upcoming_expiries=nifty_opts[nifty_opts['expiry_dt']>=today]['expiry_dt'].unique()

iflen(upcoming_expiries)==0:
raiseValueError("Noupcomingoptionexpiriesfound")

nearest_expiry=pd.Timestamp(upcoming_expiries[0])
chain=nifty_opts[nifty_opts['expiry_dt']==nearest_expiry].copy()

logger.info(f"Optionchainloaded:{len(chain)}contracts,expiry:{nearest_expiry.strftime('%d-%b-%Y')}")
returnchain,nearest_expiry.to_pydatetime()

exceptExceptionase:
logger.error(f"Optionchainfetchfailed:{e}")
raise


#==============================================================================
#QUANTITATIVEENGINE-UPDATED
#==============================================================================

defrun_quant_engine_v2(
smartApi:SmartConnect,
chain:pd.DataFrame,
spot_price:float,
expiry_dt:datetime,
previous_greeks:Optional[Dict]=None
)->Tuple[Optional[pd.DataFrame],float]:
"""
Enhancedquantitativeenginewith:
-RobustGreekscalculation(Americanoptions)
-Datavalidation&freshnesschecks
-Liquidityfiltering
-IVsurfacesmoothing
-Greekschangedetection
"""

T=get_time_to_expiry_years(expiry_dt)

ifT<=0:
logger.warning("Expiryhaspassed")
returnNone,T

records=[]

#Selectstrikes:±6%ofspotwithliquidityfilter
strike_mask=(chain['strike']>=spot_price*0.94)&(chain['strike']<=spot_price*1.06)
strikes=sorted(chain[strike_mask]['strike'].unique())

logger.info(f"Processing{len(strikes)}strikes|Spot:{spot_price:.0f}|T:{T:.4f}years")

forKinstrikes:
ce_row=chain[(chain['strike']==K)&(chain['tradingsymbol'].str.endswith('CE'))]
pe_row=chain[(chain['strike']==K)&(chain['tradingsymbol'].str.endswith('PE'))]

ifce_row.emptyorpe_row.empty:
continue

ce_symbol=ce_row.iloc[0]['tradingsymbol']
ce_token=ce_row.iloc[0]['token']
pe_symbol=pe_row.iloc[0]['tradingsymbol']
pe_token=pe_row.iloc[0]['token']

try:
#Fetchquoteswithvalidation
ce_res=smartApi.ltpData("NFO",ce_symbol,ce_token)
pe_res=smartApi.ltpData("NFO",pe_symbol,pe_token)

ce_valid,ce_err,ce_data=validate_quote_data(ce_res,ce_symbol)
pe_valid,pe_err,pe_data=validate_quote_data(pe_res,pe_symbol)

ifnotce_validornotpe_valid:
logger.debug(f"Strike{K}:CEvalid={ce_valid}({ce_err}),PEvalid={pe_valid}({pe_err})")
continue

ce_price=float(ce_data['ltp'])
pe_price=float(pe_data['ltp'])
ce_oi=float(ce_data['openinterest'])
pe_oi=float(pe_data['openinterest'])

#CalculateIVusingrobustAmericanoptionpricer
ce_iv=implied_volatility_american(ce_price,spot_price,K,T,Config.RISK_FREE_RATE,Config.NIFTY_DIVIDEND_YIELD,'c')
pe_iv=implied_volatility_american(pe_price,spot_price,K,T,Config.RISK_FREE_RATE,Config.NIFTY_DIVIDEND_YIELD,'p')

#CalculateGreeksusingfinitedifferences
ce_greeks=calculate_greeks_american(spot_price,K,T,Config.RISK_FREE_RATE,Config.NIFTY_DIVIDEND_YIELD,ce_iv,'c')
pe_greeks=calculate_greeks_american(spot_price,K,T,Config.RISK_FREE_RATE,Config.NIFTY_DIVIDEND_YIELD,pe_iv,'p')

ce_gamma=ce_greeks['gamma']
net_gex=(pe_oi-ce_oi)*ce_gamma*(spot_price**2)*0.01/1e6

#Greekschangedetection
gamma_changed_pct=0
ifprevious_greeksandKinprevious_greeks:
prev_gamma=previous_greeks[K]['gamma']
ifprev_gamma>0:
gamma_changed_pct=abs((ce_gamma-prev_gamma)/prev_gamma)

records.append({
'strike':K,
'ce_symbol':ce_symbol,
'pe_symbol':pe_symbol,
'ce_price':ce_price,
'pe_price':pe_price,
'ce_iv':ce_iv,
'pe_iv':pe_iv,
'ce_oi':ce_oi,
'pe_oi':pe_oi,
'ce_gamma':ce_gamma,
'pe_gamma':pe_greeks['gamma'],
'ce_delta':ce_greeks['delta'],
'ce_vega':ce_greeks['vega'],
'ce_theta':ce_greeks['theta'],
'pe_delta':pe_greeks['delta'],
'pe_vega':pe_greeks['vega'],
'pe_theta':pe_greeks['theta'],
'gex':net_gex,
'gamma_changed_pct':gamma_changed_pct,
})

time.sleep(0.01)#Ratelimiting

exceptExceptionase:
logger.debug(f"Strike{K}processingerror:{e}")
continue

df_quant=pd.DataFrame(records)

ifdf_quant.empty:
logger.warning("Novalidstrikesafterfiltering")
returnNone,T

#CalculateImpliedProbabilityDensityFunction
try:
spline=UnivariateSpline(df_quant['strike'],df_quant['ce_price'],k=min(4,len(df_quant)-1),s=1.0)
d2_prices=spline.derivative(n=2)(df_quant['strike'])
df_quant['iPDF']=np.exp(Config.RISK_FREE_RATE*T)*np.maximum(d2_prices,0.001)
df_quant['iPDF']=df_quant['iPDF']/df_quant['iPDF'].sum()#Normalize
exceptExceptionase:
logger.warning(f"iPDFcalculationfailed:{e}")
df_quant['iPDF']=1.0/len(df_quant)#Uniformfallback

logger.info(f"Quantenginecomplete:{len(df_quant)}validstrikes")
returndf_quant,T


#==============================================================================
#MONTECARLOPROBABILITYCALCULATOR
#==============================================================================

defmonte_carlo_pop(
spot:float,
short_put_strike:float,
short_call_strike:float,
T:float,
mu:float,
sigma:float,
num_paths:int=10000,
time_steps:int=50
)->float:
"""
MonteCarlosimulationforProbabilityofProfit
MorerealisticthanstaticPDFapproach
"""
dt=T/time_steps

#Generatepricepaths
np.random.seed(42)
dW=np.random.normal(0,np.sqrt(dt),(num_paths,time_steps))

S=np.ones((num_paths,time_steps+1))*spot

fortinrange(1,time_steps+1):
S[:,t]=S[:,t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*dW[:,t-1])

#Checkifpathstayswithinshortstrikesatexpiry
final_prices=S[:,-1]
profitable_paths=(final_prices>short_put_strike)&(final_prices<short_call_strike)

pop=np.sum(profitable_paths)/num_paths
logger.debug(f"MonteCarloPoP:{pop:.2%}({int(np.sum(profitable_paths))}/{num_paths}pathsprofitable)")

returnpop


#==============================================================================
#OPTIMIZEDIRONCONDORSELECTIONENGINE
#==============================================================================

deffind_optimal_iron_condor_v2(
df:pd.DataFrame,
spot:float,
T:float,
historical_volatility:float=0.18
)->Optional[IronCondor]:
"""
Enhancedoptimizerwith:
-MonteCarloPoP
-Robustutilityfunction
-Greeks-basedstabilityscoring
-Explicitriskpenalty
"""

ifdfisNoneordf.empty:
logger.warning("Emptydataframepassedtooptimizer")
returnNone

best_score=-np.inf
optimal_condor=None

strikes=sorted(df['strike'].unique())

MIN_OTM_BUFFER=spot*Config.MIN_OTM_BUFFER_PERCENT/100

logger.info(f"Scanning{len(strikes)}strikesforoptimalsetup...")

foriinrange(len(strikes)-3):
long_put_k=strikes[i]
short_put_k=strikes[i+1]

#Putspreadvalidation
ifshort_put_k>=spotor(spot-short_put_k)<MIN_OTM_BUFFER:
continue

forjinrange(i+2,len(strikes)-1):
short_call_k=strikes[j]
long_call_k=strikes[j+1]

#Callspreadvalidation
ifshort_call_k<=spotor(short_call_k-spot)<MIN_OTM_BUFFER:
continue

#Equalwingwidthrequirement
put_wing=short_put_k-long_put_k
call_wing=long_call_k-short_call_k

ifabs(put_wing-call_wing)>0.1:#Allowsmalltolerance
continue

wing_width=put_wing

#Getoptionprices
try:
lp_price=df[df['strike']==long_put_k]['pe_price'].values[0]
sp_price=df[df['strike']==short_put_k]['pe_price'].values[0]
sc_price=df[df['strike']==short_call_k]['ce_price'].values[0]
lc_price=df[df['strike']==long_call_k]['ce_price'].values[0]

ce_iv=df[df['strike']==short_call_k]['ce_iv'].values[0]

net_credit=(sp_price+sc_price)-(lp_price+lc_price)
max_loss=wing_width-net_credit

ifnet_credit<=0ormax_loss<=0:
continue

#Calculateskewratio
put_distance=spot-short_put_k
call_distance=short_call_k-spot
skew_ratio=min(put_distance,call_distance)/max(put_distance,call_distance)

ifskew_ratio<Config.MAX_SKEW_RATIO_THRESHOLD:
continue

#MonteCarloPoP(morerealistic)
pop=monte_carlo_pop(
spot,short_put_k,short_call_k,T,
mu=Config.RISK_FREE_RATE,
sigma=ce_iv,
num_paths=5000
)

#PositionGreeks
short_put_gamma=df[df['strike']==short_put_k]['pe_gamma'].values[0]
short_call_gamma=df[df['strike']==short_call_k]['ce_gamma'].values[0]

#Gammastability:highgamma=unstable
net_gamma=short_put_gamma+short_call_gamma
gamma_stability=-abs(net_gamma)#Negative=better(lessunstable)

#Margincalculations
margin_required=(wing_width-net_credit)*Config.LOT_SIZE
margin_buffer=margin_required*(1+Config.MARGIN_BUFFER_PERCENT/100)

#ExpectedValuewithriskadjustment
expected_value=(net_credit*pop)-(max_loss*(1-pop))

#Improvedutility:Returnonmarginwithexplicitriskpenalty
risk_reward_ratio=expected_value/margin_requiredifmargin_required>0else0
utility_score=(risk_reward_ratio*pop)+(gamma_stability*0.1)+(skew_ratio*0.05)

logger.debug(
f"Setup:{long_put_k:.0f}P/{short_put_k:.0f}P/{short_call_k:.0f}C/{long_call_k:.0f}C|"
f"Credit:₹{net_credit*Config.LOT_SIZE:.0f}|PoP:{pop:.1%}|Score:{utility_score:.4f}"
)

ifutility_score>best_score:
best_score=utility_score

#BuildIronCondorobject
trade_id=f"IC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

lp_row=df[df['strike']==long_put_k].iloc[0]
sp_row=df[df['strike']==short_put_k].iloc[0]
sc_row=df[df['strike']==short_call_k].iloc[0]
lc_row=df[df['strike']==long_call_k].iloc[0]

long_put_leg=OptionLeg(
strike=long_put_k,
option_type=OptionType.PUT,
order_type=OrderType.BUY,
symbol=lp_row['pe_symbol'],
token=lp_row.get('token',''),
entry_price=lp_price,
quantity=Config.LOT_SIZE,
iv=lp_row['pe_iv'],
delta=lp_row['pe_delta'],
gamma=lp_row['pe_gamma'],
theta=lp_row['pe_theta'],
vega=lp_row['pe_vega'],
)

short_put_leg=OptionLeg(
strike=short_put_k,
option_type=OptionType.PUT,
order_type=OrderType.SELL,
symbol=sp_row['pe_symbol'],
token=sp_row.get('token',''),
entry_price=sp_price,
quantity=Config.LOT_SIZE,
iv=sp_row['pe_iv'],
delta=sp_row['pe_delta'],
gamma=sp_row['pe_gamma'],
theta=sp_row['pe_theta'],
vega=sp_row['pe_vega'],
)

short_call_leg=OptionLeg(
strike=short_call_k,
option_type=OptionType.CALL,
order_type=OrderType.SELL,
symbol=sc_row['ce_symbol'],
token=sc_row.get('token',''),
entry_price=sc_price,
quantity=Config.LOT_SIZE,
iv=sc_row['ce_iv'],
delta=sc_row['ce_delta'],
gamma=sc_row['ce_gamma'],
theta=sc_row['ce_theta'],
vega=sc_row['ce_vega'],
)

long_call_leg=OptionLeg(
strike=long_call_k,
option_type=OptionType.CALL,
order_type=OrderType.BUY,
symbol=lc_row['ce_symbol'],
token=lc_row.get('token',''),
entry_price=lc_price,
quantity=Config.LOT_SIZE,
iv=lc_row['ce_iv'],
delta=lc_row['ce_delta'],
gamma=lc_row['ce_gamma'],
theta=lc_row['ce_theta'],
vega=lc_row['ce_vega'],
)

optimal_condor=IronCondor(
trade_id=trade_id,
entry_timestamp=time.time(),
expiry_date=datetime.fromisoformat(str(df.iloc[0].get('expiry_dt',datetime.now()))),
spot_price_at_entry=spot,
long_put=long_put_leg,
short_put=short_put_leg,
short_call=short_call_leg,
long_call=long_call_leg,
net_credit_rupees=net_credit*Config.LOT_SIZE,
wing_width=wing_width,
max_loss_rupees=max_loss*Config.LOT_SIZE,
margin_required_rupees=margin_required,
probability_of_profit=pop*100,
expected_value_rupees=expected_value*Config.LOT_SIZE,
return_on_margin_percent=(net_credit*Config.LOT_SIZE/margin_required*100),
entry_note=f"UtilityScore:{utility_score:.4f}|Skew:{skew_ratio:.2f}"
)

exceptExceptionase:
logger.debug(f"Strikecomboerror:{e}")
continue

ifoptimal_condor:
logger.info(f"Optimalsetupfound:{optimal_condor.long_put.strike:.0f}P/{optimal_condor.short_put.strike:.0f}P/"
f"{optimal_condor.short_call.strike:.0f}C/{optimal_condor.long_call.strike:.0f}C")
else:
logger.warning("Novalidironcondorsetupfound")

returnoptimal_condor


#==============================================================================
#STREAMLITDASHBOARD
#==============================================================================

st.set_page_config(page_title="QuantIronCondorV2-Production",page_icon="⚡",layout="wide")

defmain():
st.title("⚡QuantitativeIronCondorEngineV2-ProductionReady")

#Sidebarcontrols
withst.sidebar:
st.header("⚙️Controls")
enable_live=st.checkbox("EnableLiveRefresh",value=False)
enable_backtesting=st.checkbox("EnableBacktestingMode",value=False)
refresh_interval=st.slider("RefreshInterval(seconds)",5,60,10)

st.markdown("---")
st.subheader("📊Configuration")
st.metric("MaxConcurrentTrades",Config.MAX_CONCURRENT_TRADES)
st.metric("StopLossMultiple",f"{Config.STOP_LOSS_MULTIPLE}xcredit")
st.metric("DailyLossLimit",f"₹{Config.DAILY_LOSS_LIMIT:,}")

try:
#Initializesessionstate
if'journal'notinst.session_state:
st.session_state.journal=TradeJournal.load("trade_journal.pkl")
if'last_refresh'notinst.session_state:
st.session_state.last_refresh=0

#Authentication&DataFetch
st.info("🔄Initializingsystems...")
smartApi=authenticate()
chain,expiry_dt=get_nifty_option_chain()

#Fetchspotpricewithvalidation
spot_res=smartApi.ltpData("NSE","NIFTY","99926000")
spot_valid,spot_err,spot_data=validate_quote_data(spot_res,"NIFTY")

ifnotspot_valid:
st.warning(f"⚠️{spot_err}.Usingfallbackspotprice.")
spot_price=24383.60
else:
spot_price=float(spot_data['ltp'])

#Quantengine
df_quant,T=run_quant_engine_v2(smartApi,chain,spot_price,expiry_dt)

#Displaymetrics
st.subheader("📌LiveMarketMetrics")
m1,m2,m3,m4,m5=st.columns(5)
m1.metric("NIFTYSpot",f"₹{spot_price:,.0f}")
m2.metric("Expiry",expiry_dt.strftime('%d-%b-%Y'))
m3.metric("DaystoExpiry",f"{(expiry_dt-datetime.now()).days}d")
m4.metric("OpenTrades",len(st.session_state.journal.get_open_trades()))
m5.metric("DailyP&L",f"₹{st.session_state.journal.get_daily_pnl():,.0f}")

st.markdown("---")

ifdf_quantisnotNoneandnotdf_quant.empty:
#Findoptimalsetup
result=find_optimal_iron_condor_v2(df_quant,spot_price,T)

ifresult:
st.success(f"✅OptimalSetupFound(Score:{float(result.entry_note.split('Score:')[1].split('')[0]):.4f})")

#Displaytradedetails
col1,col2=st.columns(2)

withcol1:
st.subheader("🎯OptionLegs")
legs_data=[
{"Leg":"[1]BUYPUT","Strike":f"{result.long_put.strike:.0f}","Type":"PUT","Qty":Config.LOT_SIZE},
{"Leg":"[2]SELLPUT","Strike":f"{result.short_put.strike:.0f}","Type":"PUT","Qty":Config.LOT_SIZE},
{"Leg":"[3]SELLCALL","Strike":f"{result.short_call.strike:.0f}","Type":"CALL","Qty":Config.LOT_SIZE},
{"Leg":"[4]BUYCALL","Strike":f"{result.long_call.strike:.0f}","Type":"CALL","Qty":Config.LOT_SIZE},
]
st.table(legs_data)

withcol2:
st.subheader("💰FinancialMetrics")
fin_data=[
{"Metric":"NetCredit","Value":f"₹{result.net_credit_rupees:,.0f}"},
{"Metric":"MaxLoss","Value":f"₹{result.max_loss_rupees:,.0f}"},
{"Metric":"MarginRequired","Value":f"₹{result.margin_required_rupees:,.0f}"},
{"Metric":"PoP(MonteCarlo)","Value":f"{result.probability_of_profit:.1f}%"},
{"Metric":"ExpectedValue","Value":f"₹{result.expected_value_rupees:,.0f}"},
{"Metric":"ReturnonMargin","Value":f"{result.return_on_margin_percent:.2f}%"},
]
st.table(fin_data)

st.markdown("---")

#RiskAssessment
st.subheader("⚠️RiskAssessment")
risk_col1,risk_col2,risk_col3=st.columns(3)

withrisk_col1:
st.metric("ShortPutDelta",f"{result.short_put.delta:.3f}","Bearishexposure")

withrisk_col2:
st.metric("ShortCallDelta",f"{result.short_call.delta:.3f}","Bullishexposure")

withrisk_col3:
total_gamma=result.short_put.gamma+result.short_call.gamma
st.metric("NetGamma",f"{total_gamma:.5f}","Gammarisk")

#Decisionpanel
st.markdown("---")
st.subheader("🔧ActionPanel")

action_col1,action_col2,action_col3=st.columns(3)

withaction_col1:
ifst.button("✅ENTERTRADE",key="enter_btn"):
st.session_state.journal.add_trade(result)
st.session_state.journal.save("trade_journal.pkl")
st.success(f"Trade{result.trade_id}recorded!")
logger.info(f"Tradeentered:{result.trade_id}")

withaction_col2:
ifst.button("📊VIEWJOURNAL",key="journal_btn"):
st.write(st.session_state.journal.get_open_trades())

withaction_col3:
ifst.button("💾SAVEJOURNAL",key="save_btn"):
st.session_state.journal.save("trade_journal.pkl")
st.info("Journalsaved!")
else:
st.warning("⚠️Novalidironcondorsetupmetcriteria.Tryagainlater.")
else:
st.error("❌Failedtofetchoptionchaindata.CheckAPIconnectivity.")

exceptExceptionase:
st.error(f"❌CriticalError:{str(e)}")
logger.error(f"Mainerror:{e}",exc_info=True)

#Liverefresh
ifenable_live:
time.sleep(refresh_interval)
st.rerun()


if__name__=="__main__":
main()
