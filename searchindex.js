Search.setIndex({docnames:["help","index","license","lips","lips.augmented_simulators","lips.benchmark","lips.config","lips.dataset","lips.evaluation","lips.logger","lips.metrics","lips.metrics.power_grid","lips.neurips_benchmark","lips.physical_simulator","modules","quickstart"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["help.rst","index.rst","license.rst","lips.rst","lips.augmented_simulators.rst","lips.benchmark.rst","lips.config.rst","lips.dataset.rst","lips.evaluation.rst","lips.logger.rst","lips.metrics.rst","lips.metrics.power_grid.rst","lips.neurips_benchmark.rst","lips.physical_simulator.rst","modules.rst","quickstart.rst"],objects:{"":[[3,0,0,"-","lips"]],"lips.augmented_simulators":[[4,1,1,"","AugmentedSimulator"],[4,0,0,"-","hyperParameterTuner"]],"lips.augmented_simulators.AugmentedSimulator":[[4,2,1,"","build_model"],[4,2,1,"","evaluate"],[4,2,1,"","process_dataset"],[4,2,1,"","restore"],[4,2,1,"","save"],[4,2,1,"","train"]],"lips.augmented_simulators.hyperParameterTuner":[[4,1,1,"","HyperParameterTuner"]],"lips.augmented_simulators.hyperParameterTuner.HyperParameterTuner":[[4,3,1,"","augmented_simulator"],[4,2,1,"","tune"]],"lips.benchmark":[[5,0,0,"-","Benchmark"]],"lips.benchmark.Benchmark":[[5,4,1,"","__delattr__"],[5,4,1,"","__dir__"],[5,4,1,"","__eq__"],[5,4,1,"","__format__"],[5,4,1,"","__ge__"],[5,4,1,"","__getattribute__"],[5,4,1,"","__gt__"],[5,4,1,"","__hash__"],[5,4,1,"","__init_subclass__"],[5,4,1,"","__le__"],[5,4,1,"","__lt__"],[5,4,1,"","__ne__"],[5,4,1,"","__new__"],[5,4,1,"","__reduce__"],[5,4,1,"","__reduce_ex__"],[5,4,1,"","__repr__"],[5,4,1,"","__setattr__"],[5,4,1,"","__sizeof__"],[5,4,1,"","__str__"],[5,4,1,"","__subclasshook__"],[5,3,1,"id2","augmented_simulator"],[5,3,1,"id0","benchmark_name"],[5,3,1,"id4","benchmark_path"],[5,3,1,"id6","config_path"],[5,3,1,"id1","dataset"],[5,2,1,"","evaluate_simulator"],[5,3,1,"id3","evaluation"],[5,3,1,"id5","log_path"]],"lips.config":[[6,1,1,"","ConfigManager"],[6,0,0,"-","configmanager"]],"lips.config.ConfigManager":[[6,2,1,"","create_config"],[6,2,1,"","edit_config_option"],[6,2,1,"","get_option"],[6,2,1,"","get_options_dict"],[6,2,1,"","remove_config_option"],[6,2,1,"","remove_section"]],"lips.config.configmanager":[[6,1,1,"","ConfigManager"]],"lips.config.configmanager.ConfigManager":[[6,2,1,"","create_config"],[6,2,1,"","edit_config_option"],[6,2,1,"","get_option"],[6,2,1,"","get_options_dict"],[6,2,1,"","remove_config_option"],[6,2,1,"","remove_section"]],"lips.dataset":[[7,1,1,"","DataSet"],[7,1,1,"","PowerGridDataSet"],[7,0,0,"-","dataSet"],[7,0,0,"-","powergridDataSet"]],"lips.dataset.DataSet":[[7,2,1,"","generate"],[7,2,1,"","get_data"],[7,2,1,"","load"],[7,2,1,"","sample"]],"lips.dataset.PowerGridDataSet":[[7,3,1,"","ALL_VARIABLES"],[7,3,1,"","attr_names"],[7,2,1,"","extract_data"],[7,2,1,"","generate"],[7,2,1,"","get_data"],[7,2,1,"","get_sizes"],[7,2,1,"","load"],[7,3,1,"","log_path"],[7,3,1,"","name"],[7,2,1,"","reconstruct_output"],[7,2,1,"","sample"]],"lips.dataset.dataSet":[[7,1,1,"","DataSet"]],"lips.dataset.dataSet.DataSet":[[7,2,1,"","generate"],[7,2,1,"","get_data"],[7,2,1,"","load"],[7,2,1,"","sample"]],"lips.dataset.powergridDataSet":[[7,1,1,"","PowerGridDataSet"]],"lips.dataset.powergridDataSet.PowerGridDataSet":[[7,3,1,"","ALL_VARIABLES"],[7,3,1,"","attr_names"],[7,2,1,"","extract_data"],[7,2,1,"","generate"],[7,2,1,"","get_data"],[7,2,1,"","get_sizes"],[7,2,1,"","load"],[7,3,1,"","log_path"],[7,3,1,"","name"],[7,2,1,"","reconstruct_output"],[7,2,1,"","sample"]],"lips.evaluation":[[8,1,1,"","Evaluation"],[8,1,1,"","Mapper"],[8,1,1,"","PowerGridEvaluation"],[8,0,0,"-","evaluation"],[8,0,0,"-","powergrid_evaluation"],[8,0,0,"-","transport_evaluation"],[8,0,0,"-","utils"]],"lips.evaluation.Evaluation":[[8,3,1,"","INDUSTRIAL_READINESS"],[8,3,1,"","MACHINE_LEARNING"],[8,3,1,"","PHYSICS_COMPLIANCES"],[8,2,1,"","compare_simulators"],[8,3,1,"","config"],[8,3,1,"","config_path"],[8,3,1,"","config_section"],[8,2,1,"","evaluate"],[8,2,1,"","evaluate_industrial_readiness"],[8,2,1,"","evaluate_ml"],[8,2,1,"","evaluate_ood"],[8,2,1,"","evaluate_physics"],[8,2,1,"","from_benchmark"],[8,2,1,"","from_dataset"],[8,3,1,"","log_path"]],"lips.evaluation.Mapper":[[8,2,1,"","get_func"],[8,2,1,"","map_generic_criteria"],[8,2,1,"","map_label_to_func"],[8,2,1,"","map_powergrid_criteria"],[8,2,1,"","rename_key"]],"lips.evaluation.PowerGridEvaluation":[[8,2,1,"","evaluate"],[8,2,1,"","evaluate_industrial_readiness"],[8,2,1,"","evaluate_ml"],[8,2,1,"","evaluate_physics"],[8,2,1,"","from_benchmark"]],"lips.evaluation.evaluation":[[8,1,1,"","Evaluation"]],"lips.evaluation.evaluation.Evaluation":[[8,3,1,"","INDUSTRIAL_READINESS"],[8,3,1,"","MACHINE_LEARNING"],[8,3,1,"","PHYSICS_COMPLIANCES"],[8,2,1,"","compare_simulators"],[8,3,1,"","config"],[8,3,1,"","config_path"],[8,3,1,"","config_section"],[8,2,1,"","evaluate"],[8,2,1,"","evaluate_industrial_readiness"],[8,2,1,"","evaluate_ml"],[8,2,1,"","evaluate_ood"],[8,2,1,"","evaluate_physics"],[8,2,1,"","from_benchmark"],[8,2,1,"","from_dataset"],[8,3,1,"","log_path"]],"lips.evaluation.powergrid_evaluation":[[8,1,1,"","PowerGridEvaluation"]],"lips.evaluation.powergrid_evaluation.PowerGridEvaluation":[[8,2,1,"","evaluate"],[8,2,1,"","evaluate_industrial_readiness"],[8,2,1,"","evaluate_ml"],[8,2,1,"","evaluate_physics"],[8,2,1,"","from_benchmark"]],"lips.evaluation.transport_evaluation":[[8,1,1,"","TransportEvaluation"]],"lips.evaluation.transport_evaluation.TransportEvaluation":[[8,2,1,"","from_benchmark"]],"lips.evaluation.utils":[[8,1,1,"","Mapper"]],"lips.evaluation.utils.Mapper":[[8,2,1,"","get_func"],[8,2,1,"","map_generic_criteria"],[8,2,1,"","map_label_to_func"],[8,2,1,"","map_powergrid_criteria"],[8,2,1,"","rename_key"]],"lips.logger":[[9,1,1,"","CustomLogger"],[9,0,0,"-","customLogger"]],"lips.logger.CustomLogger":[[9,2,1,"","error"],[9,2,1,"","info"],[9,2,1,"","warning"]],"lips.logger.customLogger":[[9,1,1,"","CustomLogger"]],"lips.logger.customLogger.CustomLogger":[[9,2,1,"","error"],[9,2,1,"","info"],[9,2,1,"","warning"]],"lips.metrics":[[10,4,1,"","mape"],[10,4,1,"","mape_quantile"],[10,0,0,"-","ml_metrics"],[10,4,1,"","nrmse"],[10,4,1,"","pearson_r"],[11,0,0,"-","power_grid"]],"lips.metrics.ml_metrics":[[10,4,1,"","mape"],[10,4,1,"","mape_quantile"],[10,4,1,"","nrmse"],[10,4,1,"","pearson_r"]],"lips.metrics.power_grid":[[11,0,0,"-","physics_compliances"]],"lips.metrics.power_grid.physics_compliances":[[11,4,1,"","verify_current_eq"],[11,4,1,"","verify_current_pos"],[11,4,1,"","verify_disc_lines"],[11,4,1,"","verify_energy_conservation"],[11,4,1,"","verify_kcl"],[11,4,1,"","verify_loss"],[11,4,1,"","verify_loss_pos"],[11,4,1,"","verify_voltage_pos"]],"lips.physical_simulator":[[13,1,1,"","Grid2opSimulator"],[13,1,1,"","PhysicalSimulator"],[13,1,1,"","PhysicsSolver"],[13,0,0,"-","dcApproximationAS"],[13,0,0,"-","grid2opSimulator"],[13,0,0,"-","physicalSimulator"],[13,0,0,"-","physicsSolver"]],"lips.physical_simulator.Grid2opSimulator":[[13,2,1,"","get_state"],[13,2,1,"","modify_state"],[13,2,1,"","seed"],[13,2,1,"","visualize_network"],[13,2,1,"","visualize_network_reference_topology"]],"lips.physical_simulator.PhysicalSimulator":[[13,2,1,"","get_state"],[13,2,1,"","modify_state"]],"lips.physical_simulator.PhysicsSolver":[[13,2,1,"","data_to_dict"],[13,2,1,"","evaluate"],[13,2,1,"","init"],[13,2,1,"","process_dataset"],[13,2,1,"","restore"],[13,2,1,"","save"]],"lips.physical_simulator.dcApproximationAS":[[13,1,1,"","DCApproximationAS"]],"lips.physical_simulator.dcApproximationAS.DCApproximationAS":[[13,2,1,"","evaluate"],[13,2,1,"","init"],[13,2,1,"","process_dataset"],[13,2,1,"","restore"],[13,2,1,"","save"]],"lips.physical_simulator.grid2opSimulator":[[13,1,1,"","Grid2opSimulator"],[13,4,1,"","get_env"]],"lips.physical_simulator.grid2opSimulator.Grid2opSimulator":[[13,2,1,"","get_state"],[13,2,1,"","modify_state"],[13,2,1,"","seed"],[13,2,1,"","visualize_network"],[13,2,1,"","visualize_network_reference_topology"]],"lips.physical_simulator.physicalSimulator":[[13,1,1,"","PhysicalSimulator"]],"lips.physical_simulator.physicalSimulator.PhysicalSimulator":[[13,2,1,"","get_state"],[13,2,1,"","modify_state"]],"lips.physical_simulator.physicsSolver":[[13,1,1,"","PhysicsSolver"]],"lips.physical_simulator.physicsSolver.PhysicsSolver":[[13,2,1,"","data_to_dict"],[13,2,1,"","evaluate"],[13,2,1,"","init"],[13,2,1,"","process_dataset"],[13,2,1,"","restore"],[13,2,1,"","save"]],lips:[[4,0,0,"-","augmented_simulators"],[5,0,0,"-","benchmark"],[6,0,0,"-","config"],[7,0,0,"-","dataset"],[8,0,0,"-","evaluation"],[9,0,0,"-","logger"],[10,0,0,"-","metrics"],[13,0,0,"-","physical_simulator"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[2,4,6,7,8,10,11,13],"0003":4,"001":11,"1":[4,10,11],"10":10,"128":4,"150":4,"2":[6,7,8,10,11,13],"2021":[6,7,8,10,13],"2022":[7,13],"3":[4,11,15],"30":2,"3e":4,"4":[4,11],"5":4,"50":2,"6":15,"60":2,"abstract":[5,13],"byte":5,"case":[2,7,11,13],"class":[4,5,6,7,8,9,13],"default":[4,5,6,7,8,10,11,13],"do":[2,8,10],"final":[2,7,8],"float":[10,11],"function":[1,4,5,6,7,8,10,11,13],"import":[2,7],"int":[4,7,13],"new":[5,8,11],"null":11,"public":[2,6,7,8,10,13],"return":[4,5,6,7,8,10,11,13],"true":[5,7,8,10],A:[4,7,11,13],And:13,For:[2,7,10,13],If:[0,2,5,6,7,8,10,13],In:[2,7,13,15],It:[4,5,7,8,10,11,13],Its:10,No:2,Not:7,Such:2,The:[2,5,7,8,10,11,13,15],Then:10,There:7,These:[4,10,11,13],To:11,__delattr__:5,__dir__:5,__eq__:5,__format__:5,__ge__:5,__getattribute__:5,__gt__:5,__hash__:5,__init__:6,__init_subclass__:5,__le__:5,__lt__:5,__ne__:5,__new__:5,__reduce__:5,__reduce_ex__:5,__repr__:5,__setattr__:5,__sizeof__:5,__str__:5,__subclasscheck__:5,__subclasshook__:5,_attr_i:13,_description_:[8,13],a_ex:[7,11],a_or:[7,11],abadi:0,abc:[4,5],abcmeta:5,abil:2,abl:[2,7,8,13,15],about:11,abov:[2,6,7,10],absenc:2,absolut:[2,10],abstractmethod:[4,5],accept:13,accur:[2,5],action:[2,13],activ:[4,15],actor:[7,13],actor_se:7,actor_typ:13,ad:[6,11],add:[2,6,8,13],addit:5,addition:2,affect:2,affero:2,after:2,against:2,agent:[7,13],aggreg:10,agre:2,agreement:2,algorithm:5,all:[1,2,4,6,7,8,11,13],all_vari:7,alleg:2,allow:[2,4,5,8,13],almost:10,alon:2,alreadi:[6,8],also:[2,7],alter:2,alwai:10,amongst:7,an:[2,4,5,6,8,10,11,13],angl:11,ani:2,anoth:13,anyon:2,anyth:[4,13],apparatu:2,appli:2,applic:7,appropri:5,ar:[1,2,4,7,8,10,11,13],arg:5,argument:11,arrai:[7,11],assert:2,assum:2,attach:2,attempt:2,attr_nam:7,attribut:[5,7,8],augment:[4,5,8,11,13],augmented_simul:[1,3,5,14],augmentedsimul:[1,3,5,8,13,14],augmnet:5,author:[2,6,7,8,10,13],automat:2,avail:[2,8],averag:[10,11],avoid:10,back:2,base:[4,5,6,7,8,9,13],baseact:13,baseag:[7,13],basi:2,basic:8,batch1:7,batch2:7,batch:[4,7],batch_siz:4,bdonnot:10,becom:2,been:[2,7,10],befor:5,behalf:2,behaviour:[4,13],believ:2,bellow:10,benchmark1:[1,3,13,14],benchmark2:[1,3,14],benchmark3:[1,3,14],benchmark:[1,3,6,7,8,10,13,14],benchmark_nam:[5,13],benchmark_path:5,benefici:2,between:[8,10,11,13],bin:15,blob:10,bonu:8,bool:[4,7],both:[8,11,13],bring:2,brought:2,build:4,build_model:4,built:[4,13],busi:2,c:[6,7,8,10,13],cach:5,call:[5,7,11,13],callabl:[7,8],can:[2,4,5,6,7,8,10,13],cannot:[4,7],capac:8,categori:8,caus:2,cd:15,charact:2,charg:2,check:11,check_solut:11,checkout:15,choic:2,choos:2,chronic:13,chronics_selected_regex:13,circumst:2,classmethod:8,clear:2,clone:15,code:[6,7,8,10,13],coeffici:10,column:10,com:[10,15],combin:2,come:[2,4,7,13],comment:7,commerci:2,common:2,compare_simul:8,compat:13,complet:[2,10],complex:7,complianc:[2,8],compliant:2,compon:10,compris:[8,11],comput:[2,8,10,11,13],concat:7,concaten:7,concern:2,config:[1,3,5,7,8,11,14],config_path:[5,8,13],config_sect:8,configmanag:[1,3,7,8,14],configpars:6,conflict:2,consecut:7,consequenti:2,conserv:[8,11],consid:[10,11],consist:10,constant:10,constitut:2,constru:2,constructor:13,contact:0,contain:[2,5,11],content:2,context:8,contract:2,control:2,convei:2,converg:13,copi:[2,6,7,8,10,13],copyright:[2,6,7,8,10,13],correct:2,correctli:10,correl:10,correspond:[5,7,8,10,13],cost:2,could:[4,8],count:7,counter:2,court:2,creat:[2,4,5,6,7,8,13],create_config:6,creation:2,criteria:[5,8],critiera:8,cross:[2,4],current:[7,8,11],custom:5,customlogg:[1,3,14],dai:2,damag:2,data:[4,5,7,8,11,13],data_to_dict:13,databas:7,dataset:[1,3,4,5,8,13,14],dc_approxim:13,dcapproximationa:[1,3,14],deactiv:10,deal:2,death:2,declaratori:2,defect:2,defend:2,defin:2,definit:7,delattr:5,delet:2,depend:15,describ:2,descript:[2,8,13],desir:2,detail:2,dev:15,dict:[4,5,7,8,11,13],dictionari:[5,6,7,8,11,13],dictionnari:13,differ:[2,7,10,13],dir:5,direct:2,directli:2,directori:[2,7],disc_lin:8,disconnect:11,displai:2,distinguish:2,distribut:[6,7,8,10,13],distributor:2,divid:10,doctrin:2,document:[0,2],doe:[2,5,6,11],domain:[10,13],done:13,doubl:13,drafter:2,e:15,each:[2,8,10,11],earli:5,earlier:2,eas:6,edit:6,edit_config_opt:6,either:[2,7],el:11,electr:[8,11],element:[7,10],empti:[5,13],emul:[4,13],en:[6,8,10],end:2,energi:[8,11],enforc:2,entir:[2,7],entiti:2,env:[11,13],env_kwarg:13,environ:[7,11,13],ep:11,episod:13,epoch:4,eq:8,equal:10,equat:[11,13],equival:2,error:[4,9,10],essenti:2,evalu:[1,3,4,5,11,13,14],evaluate_industrial_readi:8,evaluate_ml:8,evaluate_ood:8,evaluate_phys:8,evaluate_simul:5,evalut:5,even:2,event:2,everi:2,everywher:10,evolut:13,exact:7,exampl:[4,7,8,10,13],except:[2,4],exclud:2,exclus:2,exercis:2,exist:[6,8],expect:[4,7],experi:[5,7,13,15],experiment_nam:7,explain:1,explicitli:[2,11],exploit:2,express:[2,11],extend:5,extent:2,extra:[6,13],extract:7,extract_data:7,extrem:11,factual:2,fail:[2,7,11],failed_indic:11,failur:2,fals:5,featur:10,fee:2,fifti:2,file:[0,2,5,6,7,8,10,11,13],fine:4,first:[2,7,13],fit:2,flow:13,fold:4,folder:15,follow:[2,11,13,15],form:[6,7,8,10,13],format:6,format_spec:5,formatt:5,found:[7,8],foundat:2,fr:[0,6,7,8,10,13],framework:[6,7],free:[2,7],from:[2,4,6,7,8,10,13],from_benchmark:8,from_dataset:8,full:[4,13],fullyconnecteda:[1,3,14],func:8,further:2,futur:11,gener:[2,7,8,11],generate_data:7,get:[7,8],get_data:7,get_env:13,get_func:8,get_opt:6,get_options_dict:6,get_result:13,get_siz:7,get_stat:13,getattr:5,getter:13,git:15,github:[10,15],give:[10,11,13],given:[2,4,8,10,11,13],global:11,gnu:2,goal:13,goodwil:2,govern:2,greater:11,gri2op:11,grid2op:[7,11,13],grid2opsimul:[1,3,7,14],grid:[4,8,13],grid_path:13,ground:11,guarante:7,guid:0,ha:[2,7,10],happen:10,hash:5,have:[0,2,7,13],held:2,help:[1,5],helper:[5,8],henc:11,hereaft:2,herebi:2,hereof:2,higher:7,highest:10,how:[0,2,11,15],howev:2,http:[2,6,7,8,10,13,15],hyper:4,hyperparametertun:[1,3,14],i:2,ideal:13,identifi:[6,7,8,10,13],ignor:10,ii:2,illustr:13,implement:[5,7,8,10,11,13],impli:2,imposs:[2,7],inaccuraci:2,incident:2,includ:[2,5,8,11],incur:2,indemn:2,indemnifi:2,independ:10,index:[1,7],indic:[5,11],indirect:2,indirectli:2,individu:2,indr:8,industri:[1,8],industrial_readi:8,infer:8,info:9,inform:[2,10,11,13],infring:2,init:[10,11,13],initi:[2,4,5,7,8,13],initial_chronics_id:13,injuri:2,input:7,instabl:10,instal:[1,7],instanc:8,instead:10,integ:[7,13],intellectu:2,intend:2,interest:10,interfac:[0,7],intern:13,intial:8,invok:5,involv:7,irrelev:7,irt:[0,6,7,8,10,13],issu:7,issubclass:5,iter:[4,11],its:[2,13,15],job:4,joul:8,judgment:2,judici:2,jurisdict:2,just:7,k_new:8,k_old:8,kcl:11,kcl_valu:11,keep:[10,13],kei:[7,8,11,13],kept:10,kind:[2,7],kirchhoff:[8,11],known:2,kwarg:[4,5,6,11,13],label:8,languag:2,last:15,later:2,law:[2,8,11],layer:4,layer_act:4,lce:11,leap_net:10,leapnet:4,leapneta:[1,3,14],learn:[1,4,8,13],least:[7,8],legal:2,lesser:2,level:11,leyli:0,liabl:2,licenc:[6,7,8,10,13],licens:[1,6,7,8,10,13],lightsim2grid:11,like:[2,4],line:[7,11],line_statu:7,linear:13,lip:15,list:[4,6,7,8,11],load:[7,11],load_p:7,load_q:7,load_v:7,locat:[2,4],log:[7,8,11,13],log_path:[4,5,7,8,9,11],logger:[1,3,5,13,14],logo:2,look:[2,7],loss:[2,4,8,11],lost:2,lr:4,m:15,machin:8,machine_learn:8,made:[2,11],mae:8,mai:[2,5,11],main:[8,13],maintain:[2,8],make:[2,7],malfunct:2,manag:[2,7],mani:10,manner:2,map:8,map_generic_criteria:8,map_label_to_func:8,map_powergrid_criteria:8,mape90:8,mape:10,mape_quantil:10,mapper:8,mark:2,master:10,match:[7,10],materi:2,matric:10,matter:2,max:10,maximum:2,mean:[2,7,10],meant:[4,13],memori:5,merchant:2,messag:9,meta:8,metadata:7,method:[2,4,5,6,7,8,9,13],metric:[1,3,5,8,14],might:[7,13],milad:0,min:10,minimalist:8,ml:[8,15],ml_metric:[1,3,14],mleyliabadi:15,model:[4,13],modifi:[7,8,13],modify_st:13,modul:[1,3,14],more:[2,7,10],moreov:2,mozilla:[2,6,7,8,10,13],mpl:[2,6,7,8,10,13],mse:[4,8],multioutput:10,multipl:[5,8],must:2,my:15,n_fold:4,n_job:4,name:[2,4,5,7,8,9,13],nan:10,nb:7,nb_iter:4,nb_sampl:7,ndarrai:[7,10],necessari:2,need:11,neg:7,neglig:2,network:[4,6,7,8,10,13],neural:[4,13],neurips_benchmark:[1,3,14],nn:4,non:[2,7,10],none:[4,5,6,7,8,9,11,13],norm:10,normal:[5,10],note:[2,8,10],noth:[2,5],notic:8,notifi:2,notimpl:5,notwithstand:2,now:[7,13],np:[7,10],nrmse:10,number:[2,4,7,10],numer:10,numpi:[7,10],object:[4,5,6,7,8,9,13],oblig:2,observ:[8,11,13],obtain:[2,6,7,8,10,13],offer:[2,7,11],old:8,onc:13,one:[2,4,6,7,8,10,13],one_exampl:13,ongo:2,onli:[2,10,11],ood:8,option:[2,4,5,6,7,8,9,11,13],order:[2,13],ordinari:2,org:[2,6,7,8,10,13],origin:2,other:[2,7],otherwis:[2,5,7,10],out:8,outcom:5,output:7,outstand:2,over:[4,10,11,13],overload:5,overrid:5,overridden:[5,8],own:2,ownership:2,p:11,p_ex:[7,11],p_or:[7,11],packag:[1,14,15],page:[1,15],parallel:4,paramet:[4,5,6,7,8,10,11,13],parameter:5,parser:6,part:[2,6,7,8,10,13],parti:2,particular:2,pass:5,path:[4,5,6,7,8,11,13],path_out:[7,13],pathlib:4,pearson:10,pearson_r:10,percent:2,percentag:[10,11],perform:[2,4,5,13],permit:2,person:2,phase:4,physic:[1,5,8,13],physical_simul:[1,3,5,7,14],physicalsimul:[1,3,5,7,14],physics_compli:[3,8,10],physicssolv:[1,3,5,14],pickl:5,pip3:15,place:2,plain:7,plateform:2,platform:[1,6,7,8,10,13,15],pleas:0,plot_kwarg:13,plotgrid:13,plotmatplot:13,point:[5,8,10],popul:8,portion:2,posit:[8,11],possibl:[2,7,11],power:[2,6,7,8,10,11,13],power_grid:[3,10],powergird:7,powergrid:[7,8,13],powergrid_evalu:[1,3,14],powergridbenchmark:[1,3,8,14],powergriddataset:[1,3,13,14],powergridevalu:8,predefin:5,predict:[5,8,10,11,13],prefer:2,present:[2,13,15],prevent:2,previou:10,previous:7,princip:2,prior:2,process:[2,13],process_dataset:[4,13],prod_p:[7,11],prod_q:7,prod_v:7,product:11,profit:2,prohibit:2,project:15,properti:2,proport:11,protocol:5,prove:2,provid:[2,4,8],provis:2,provision:2,publish:2,purpos:2,put:[2,10],py:[6,10],python3:15,python:[6,7,8,10,13,15],q:[10,11],q_ex:[7,11],q_or:[7,11],qualiti:2,quantil:10,rais:[4,7,13],random:7,rate:4,ratio:10,readi:8,real:7,reason:2,receipt:2,receiv:2,recipi:2,recommend:15,reconstruct:7,reconstruct_output:7,reconstrut:7,ref_ob:11,refer:[2,7,11,13],reform:2,regist:8,reinstat:2,relat:[2,11],relev:2,reload:8,relu:4,remedi:2,remov:[2,6,7,13],remove_config_opt:6,remove_sect:6,renam:[2,8],rename_kei:8,repair:2,replac:[7,10,11],report:[8,11],repositori:[7,15],repr:5,repres:[2,7,13],reproduc:[2,7,13],request:5,requir:[1,2,5,7,8,11],resel:2,respect:[2,5,8,11],restor:[4,13],restrict:2,result:[2,4,5,7,8,11,13],retriev:7,retrun:6,right:[2,7],risk:2,rmse:10,root:10,routin:10,row:[4,7,10,13],royalti:2,rte:[7,13],run:[4,15],runtimeerror:[7,13],s:[2,8,11,13],sale:2,sampl:7,sampler:7,save:[4,5,7,8,11,13],save_path:[5,8],scalabl:8,scalar:11,scenario:[6,8],scenario_nam:6,scikit:8,search:[1,4],secondli:13,section:[2,6,8,10],section_nam:6,see:[2,5,6,7,8,10,13],seed:[7,13],sef:7,self:[5,7,13],sell:2,separ:2,serv:[1,7,13],servic:2,set:[4,7,10,11,13],setattr:5,shall:2,shape:10,share:2,should:[2,4,5,7,8,10,11,13,15],show:[13,15],side:8,signatur:5,simul:[1,4,5,7,8,11,13],simulation_nam:7,simulator_se:7,singl:[5,7,10],size:[4,5,7],size_i:7,size_tau:7,size_x:7,sizes_lay:4,skill:2,so:2,solv:13,solver:13,some:[2,4,7,10,13],someth:[4,7,13],sourc:[4,5,6,7,8,9,10,11,13],spdx:[6,7,8,10,13],special:2,specif:[2,7,8,10,13],sqrt:11,squar:10,state:[4,13],statutori:2,step:[7,11,13,15],steward:2,stoppag:2,store:[7,8,11],str:[4,5,6,7,8,9,10,11,13],strict:13,subclass:[5,8,13],subject:[2,6,7,8,10,13],sublicens:2,submodul:[1,3,14],subpackag:[1,14],substanc:2,substat:11,suffici:2,supplementari:11,support:2,suppos:10,surviv:2,system:[10,13],systemx:[0,6,7,8,10,13],tak:8,take:[5,7,10],term:[6,7,8,10,13],test:[5,7,8],text:[2,7],than:[2,7,11],thei:[2,4,7,13],them:[7,8],theori:2,thereof:2,theta:[7,11],theta_ex:7,theta_or:7,thi:[0,1,2,4,5,6,7,8,10,11,13,15],third:2,those:2,threshold:10,thu:10,time:[2,8,13],todo:8,tol:11,toler:11,topo_vect:7,topolog:13,torch:4,tort:2,trademark:2,train:[4,5,7,8],train_dataset:4,transfer:2,transform:[4,13],transport_evalu:[1,3,14],transportevalu:8,troubl:0,truth:11,tune:4,tupl:[4,7,11,13],two:[8,10,11,13],txt:[6,7,8,10,13],type:[4,5,7,8,10,11,13],u:15,under:[2,6],understand:2,unenforc:2,uniform:[7,10],uniformli:7,union:[4,5,7,8,10,13],unless:2,unmodifi:2,until:[2,13],unus:7,updat:[8,11],us:[0,4,5,6,7,8,10,11,13],usag:[6,7,8,10,11,13],usecas:7,user:[0,1,2,15],util:[1,3,14],v:[2,11],v_ex:[7,11],v_or:[7,11],val_dataset:4,valid:[4,5,7],validli:2,valu:[5,6,7,10,11],variabl:[7,8,10,11],variou:[4,5,8],vector:10,venv_lip:15,verbos:4,verif:[8,11],verifi:[8,11],verify_current_eq:11,verify_current_po:11,verify_disc_lin:11,verify_energy_conserv:11,verify_kcl:11,verify_loss:11,verify_loss_po:11,verify_voltage_po:11,version:[6,7,8,10,13],view:[5,8],violat:11,violation_percentag:11,violation_prop_node_level:11,violation_prop_obs_level:11,visual:13,visualize_network:13,visualize_network_reference_topolog:13,voltag:[8,10,11],wa:[2,6,7,8,10,13],wai:11,want:[2,4,13],warn:9,we:[7,10,15],weher:11,when:[5,7,10],where:[2,4,7,8,10,11,13],whether:[2,10],which:[2,4,5,7,8,10,11,13],who:2,whole:13,wide:2,wise:10,within:2,without:2,world:[2,7],would:2,www:[6,7,8,10,13],x:7,y:7,y_pred:10,y_true:10,you:[0,4,6,7,8,10,13],zero:11},titles:["Help","Welcome to LIPS\u2019s documentation!","License","lips package","lips.augmented_simulators package","lips.benchmark package","lips.config package","lips.dataset package","lips.evaluation package","lips.logger package","lips.metrics package","lips.metrics.power_grid package","lips.neurips_benchmark package","lips.physical_simulator package","lips","Getting started"],titleterms:{"1":2,"10":2,"11":2,"12":2,"13":2,"14":2,"2":2,"3":2,"4":2,"5":2,"6":2,"7":2,"8":2,"9":2,"new":2,A:2,To:15,With:2,addit:2,applic:2,augmented_simul:4,augmentedsimul:4,b:2,benchmark1:12,benchmark2:12,benchmark3:12,benchmark:5,claim:2,code:2,compli:2,condit:2,config:6,configmanag:6,contribut:[2,15],contributor:2,cover:2,creat:15,customlogg:9,dataset:7,date:2,dcapproximationa:13,definit:2,disclaim:2,distribut:2,document:1,due:2,effect:2,enter:15,environ:15,evalu:8,execut:2,exhibit:2,fair:2,form:2,from:15,fullyconnecteda:4,get:[1,15],grant:2,grid2opsimul:13,guid:1,help:0,hyperparametertun:4,inabl:2,incompat:2,indic:1,instal:15,larger:2,leapneta:4,liabil:2,licens:2,limit:2,lip:[1,3,4,5,6,7,8,9,10,11,12,13,14],litig:2,logger:9,metric:[10,11],miscellan:2,ml_metric:10,modif:2,modifi:2,modul:[4,5,6,7,8,9,10,11,12,13],neurips_benchmark:12,notic:2,option:15,packag:[3,4,5,6,7,8,9,10,11,12,13],patent:2,physical_simul:13,physicalsimul:13,physics_compli:11,physicssolv:13,power_grid:11,powergrid_evalu:8,powergridbenchmark:5,powergriddataset:7,regul:2,represent:2,requir:15,respons:2,s:1,scope:2,secondari:2,setup:15,softwar:2,sourc:[2,15],start:[1,15],statut:2,submodul:[4,5,6,7,8,9,10,11,12,13],subpackag:[3,10],subsequ:2,tabl:1,technic:1,term:2,termin:2,todo:[7,11,13],transport_evalu:8,us:2,util:8,version:2,virtual:15,virtualenv:15,warranti:2,welcom:1,work:2,you:2,your:2}})