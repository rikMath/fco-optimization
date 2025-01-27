import pandas as pd
import json
from pyomo.environ import *
# custos de transporte
with open("dados/CustosTransp.json", "r") as file:
    custos_transporte = json.load(file)

df_custos = pd.DataFrame(custos_transporte.items(), columns=["Chave", "Custo_Transporte"])
df_custos[['Depósito_Origem', 'Depósito_Destino', 'Material']] = df_custos['Chave'].str.extract(r'\((\d+),\s*(\d+),\s*(\d+)\)')
df_custos = df_custos.drop(columns=["Chave"])
df_custos = df_custos[["Depósito_Origem", "Depósito_Destino", "Material", "Custo_Transporte"]]
df_custos = df_custos.astype({"Depósito_Origem": int, "Depósito_Destino": int, "Material": int, "Custo_Transporte": float})

# estoque
df_estoque = pd.read_excel("dados/Estoque.xlsx")

# obras
df_obras = pd.read_excel("dados/Obras.xlsx")
# Conjuntos
dep_orig = df_custos['Depósito_Origem'].unique()  # Depósitos de origem
dep_dest = df_custos['Depósito_Destino'].unique()  # Depósitos de destino
materiais = df_estoque['COD_MAT'].unique()  # Materiais
obras = df_obras['OBRA'].unique()  # Obras

prioridades = df_obras.set_index('OBRA')['PRIOR'].to_dict()

# Estoque disponível
estoque = df_estoque.set_index(['COD_DEP', 'COD_MAT'])['ESTOQ'].to_dict()

# Custos de transporte
custos_transp = df_custos.set_index(['Depósito_Origem', 'Depósito_Destino', 'Material'])['Custo_Transporte'].to_dict()
custos_transp = {(int(k), int(j), int(m)): v for (k, j, m), v in custos_transp.items()}

# Demandas das obras
demandas = df_obras.set_index(['OBRA', 'COD_MAT'])['QTD_DEM'].to_dict()

# Store valid keys for faster checks
valid_indices = {
    (k, j, m): v for (k, j, m), v in custos_transp.items()
    if (k, m) in estoque and estoque[(k, m)] > 0 and
       any(demandas.get((o, m), 0) > 0 for o in obras)
}
# Convert estoque to DataFrame
df_estoque_dense = pd.DataFrame(list(estoque.items()), columns=['Index', 'Value'])
df_estoque_dense[['Dep', 'Mat']] = pd.DataFrame(df_estoque_dense['Index'].tolist(), index=df_estoque_dense.index)
df_estoque_dense = df_estoque_dense.pivot(index='Dep', columns='Mat', values='Value').fillna(0)

# Convert demandas to DataFrame
df_demandas_dense = pd.DataFrame(list(demandas.items()), columns=['Index', 'Value'])
df_demandas_dense[['Obra', 'Mat']] = pd.DataFrame(df_demandas_dense['Index'].tolist(), index=df_demandas_dense.index)
df_demandas_dense = df_demandas_dense.pivot(index='Obra', columns='Mat', values='Value').fillna(0)
model = ConcreteModel()

# Conjuntos
model.D_orig = Set(initialize=dep_orig)  # Depósitos de origem
model.D_dest = Set(initialize=dep_dest)  # Depósitos de destino
model.M = Set(initialize=materiais)  # Materiais
model.O = Set(initialize=obras)  # Obras

# Define filtered sets
model.D_filtered = Set(initialize={j for (_, j, _) in valid_indices})
model.M_filtered = Set(initialize={m for (_, _, m) in valid_indices})
model.KM_filtered = Set(dimen=2, initialize={(k, m) for (k, _, m) in valid_indices})

# Parâmetros
model.prioridade = Param(model.O, initialize=prioridades, within=NonNegativeReals)  # Prioridades das obras
model.estoque = Param(model.D_orig * model.M, initialize=estoque, default=0)  # Estoques nos depósitos
model.demandas = Param(model.O * model.M, initialize=demandas, default=0)  # Demandas das obras
model.custos_transporte = Param(model.D_orig * model.D_dest * model.M, initialize=valid_indices, default=float('inf'))  # Custos de transporte

model.x = Var(model.O, domain=Binary)  # x[i] indica se a obra i está alocada
model.t = Var(model.D_orig, model.D_dest, model.M, domain=NonNegativeReals)  # Transporte de materiais entre depósitos
@model.Objective(sense=maximize)
def obj1(model):
    return sum(model.x[i] * model.prioridade[i] for i in model.O)

@model.Objective(sense=minimize)
def obj2(model):
    return sum(model.t[k, j, m] * model.custos_transporte[k, j, m] for k, j, m in valid_indices.keys())

@model.Constraint()
def restricao_obra_unica(model):
    return sum(model.x[i] for i in model.O) <= 1

@model.Constraint(model.D_filtered, model.M_filtered)
def restricao_demanda(model, k, m):
    return (
        sum(model.demandas[i, m] * model.x[i] for i in model.O) +
        sum(model.t[k, jj, mm] for kk, jj, mm in valid_indices.keys()) - 
        sum(model.t[jj, k, mm] for kk, jj, mm in valid_indices.keys()) <=
        model.estoque[k, m]
        
    )

#@model.Constraint(model.D_orig, model.M)
#def restricao_demanda(model, k, m):
#    return (
#        sum(model.demandas[i, m] * model.x[i] for i in model.O) +
#        sum(model.t[k, j, m] for j in model.D_dest) <=
#        model.estoque[j, m] +
#        sum(model.t[k, j, m] for j in model.D_dest)
#    )
opt = SolverFactory('highs', executable=r"C:\Users\DárcioMeloBragançaSi\OneDrive - DetronicEnergia\Downloads\highs.exe")
model.obj1.deactivate()
model.obj2.activate()
results = opt.solve(model, tee=True)

opt = SolverFactory('highs', executable=r"C:\Users\DárcioMeloBragançaSi\OneDrive - DetronicEnergia\Downloads\highs.exe")
model.obj1.activate()
model.obj2.deactivate()
results = opt.solve(model, tee=True)