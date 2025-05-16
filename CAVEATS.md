# Caveats

1. **NVEXT Evaluation Limitations**: The system cannot evaluate any functionality that utilizes NVEXT, except when NVEXT is solely used for metadata purposes. This means that any core logic or processing relying on NVEXT will not be evaluated.

2. **Single Choice Support**: The service currently supports only a single choice in responses. This implies that if a response contains multiple choices, only the first one will be processed and evaluated.

3. **Text-Only Modality**: The system is not multimodal and only supports text-based inputs and outputs. Any data requiring audio, video, or other non-textual formats cannot be processed.

4. **Parallel Tool Calling**: The service does not support parallel tool calling. Each tool call must be executed sequentially, which may impact performance in scenarios requiring simultaneous tool interactions.

5. **Dataset Validation**: There is no dataset validation implemented yet. This means that datasets are assumed to be correct and complete, and any errors or inconsistencies in the data will not be automatically detected or flagged. Many of the thigns in this list are candidates for pre-job validation.

6. **NIM Launch via DMS**: The system does not support launching NIMs via the Data Management System (DMS). Any attempts to deploy NIMs using DMS will not be successful.

7. **Customization Limitations**: The service does not perform any customization of models or configurations. All operations are executed with predefined settings, and there is no capability to adjust or tailor these settings to specific needs or preferences.

8. **Limited Evaluation Metrics**: The current evaluation system only calculates a similarity score as a metric. It does not support additional metrics, which could provide a more comprehensive evaluation of model performance. Implementing other metrics could enhance the evaluation process and is worth considering, given the relatively low cost of doing so.

9. **Prompt-Tuned Model Limitations**: The system cannot utilize prompt-tuned models. This limitation arises because the build.nvidia.com endpoints do not support prompt-tuned models. Additionally, only higher versions of NIM support these models, meaning that even downloaded versions of NIM cannot accommodate prompt-tuned models.

10. **Rate Limiting with NVIDIA Judge**: When using the build.nvidia.com service as our judge, particularly with the 3.3 70b model, we may easily encounter rate limit exceeded errors. This is due to the high demand and limited capacity of the service, which can restrict the number of requests processed in a given timeframe. Users should be aware of this limitation and plan their evaluations accordingly to avoid disruptions.

11. **Can't see inference outputs**: Not possible to debug issues in models that have inference outputs

12. **Generic judge prompt**: Might want to expose an optional judge prompt for the workload POST jobs endpoint

13. **Customizer stops on validation loss**: There is no way to force running a certain number of epochs. There is no way to control which checkpoint gets uploaded. https://nvidia.slack.com/archives/C06FAGGGZ0E/p1745269398073679 -- big bummer for flywheel. will need to WAR.


Cannot use platform due to NFS lock issues. Womp. https://nvidia.slack.com/archives/C06FAGGGZ0E/p1745548649082089?thread_ts=1745548197.903749&cid=C06FAGGGZ0E


## DMS

- Doesn't warn if name is too long
- Keeps saying `pending` even after the deployment fails due to name being too long

## NIM

- Doesn't propagate metadata to logging proxy
- would be nice to know deployment info on nim proxy too: image and tag


## Customizer

- Let us know when training container is downloading
- tell us iteration % done not just epoch % done


## FO

- check platform status before starting
- check customizer supported models
- change running status to "Waiting for model sync" once customization is done
- add the hash for datasets and customizations efficiency


## Evaluator

- latency on NIM & judge requests? want to know if getting rate limited
- can we get scores during testing? be nice to see what's happening!
- api endpoint should have headers so i can pass things like Host, x-[stuff] etc...
