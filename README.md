### Input data

* transactional data (`user_id`, `ts`, `event`, `value`)
* user attributes (`user_id`, ...)
* profit margins

### Pipeline

1. Clean / aggregate data
2. Split event data
3. Construct user histories
4. Fit scaling parameters
5. Construct batches for training
6. Train
7. Evaluate