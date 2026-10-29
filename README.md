# CLARO GUIA

Projeto acadêmico do Bootcamp Claro que simula o consumo de internet e recomenda planos. Usa Random Forest para prever a demanda e um recomendador híbrido de Random Forest e filtragem colaborativa por Pearson.

**Dados, planos e histórico de ofertas são fictícios.** As métricas avaliam a simulação, sem representar desempenho com clientes reais.

## Executar

Instale o Pixi:

**macOS/Linux**

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Abra um novo terminal e execute na pasta do projeto:

```bash
pixi install
pixi run ver-site
```

Abra [localhost:8000](http://localhost:8000). Para encerrar, pressione **Ctrl + C** no terminal.

| Comando | Finalidade |
|---|---|
| `pixi run editar` | Editar e executar o notebook Marimo completo. |
| `pixi run site` | Gerar o site estático. |
| `pixi run testes` | Executar os testes após gerar o site. |

## Documentação

[Detalhes sobre o projeto](http://localhost:8000/detalhes_do_projeto.html) centraliza funcionamento, protocolo científico, métricas e referências. O link local fica disponível após iniciar o site.

O site inclui um notebook editável no navegador, com opção de restaurar a versão original. As edições dos visitantes não alteram o repositório.

O [workflow](.github/workflows/publicar_site.yml) gera o site e executa os testes antes de publicar. Novos envios para a branch principal repetem esse processo.
