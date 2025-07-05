# commandの扱われ方
command_managerはManagerBasedEnvが持つ。これは、step()の中でcommand_manager.compute()を呼ぶ。
コードのどこに該当部分があるのかは分からないが、訓練を動かした時はManagerBasedRLEnv::load_managersは起動時に呼ばれる。load_managerの中で、command_managerは実体化される。

manager_based_rl_env.py内
```python
    cfg: ManagerBasedRLEnvCfg
    self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
```

## CommandManager
_prepare_termsがtermを実体化するもの。
cfg.class_typeを実体化するようだ。ということは、CommandTermを作って、それをclass_typeに持つConfigを書けばいいのか.
以下の_prepare_terms()はsuperクラスであるManagerBaseの__init__内で呼ばれる。

```python
    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, CommandTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CommandTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)  # ここでtermを実体化してそう
            # sanity check if term is valid type
            if not isinstance(term, CommandTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type CommandType.")
            # add class to dict
            self._terms[term_name] = term

```


class_typeは普通にクラスを指定してるだけだった。ちなみにUniformVelocityCommandはCommandTermを継承している。

```python
@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand
```


termの実行方法
command_manager.compute()が呼ばれる。その中では、termがそれぞれ呼ばれてそれのcomputeが呼ばれる
```python
    # iterate over all the command terms
    for term in self._terms.values():
        # compute term's value
        term.compute(dt)
```

termのcomputeは以下の通り

```python
    # Term.compute()
    def compute(self, dt: float):
        # update the metrics based on current state
        self._update_metrics()
        # reduce the time left before resampling
        self.time_left -= dt
        # resample the command if necessary
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        # update the command
        self._update_command()
```


## stepで呼ばれるもの
command_manager.compute()が毎step呼ばれる。
それでは、全てのtermのcomputeが呼ばれる。computeは上のやつが呼ばれる。
その後、observationの中にgenerated_commandsがあれば、そいつの中からterm.command()が呼ばれる。
それでは、vel_command_bが直に返される。