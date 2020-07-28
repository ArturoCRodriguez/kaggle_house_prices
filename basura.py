for x in range(1,X.shape[1]):        
    count = 0
    finals = []    
    # for i in X_test_prepared_df.columns:
    #     if x > (len(finals) + len(removed)) and i not in removed:
    #         last_added_column = i
    #         finals.append(last_added_column)
    #         count += 1
    # for i in r.importances_mean.argsort()[::-1]:
    #     if count < (x - len(removed)) and X.columns[i] not in removed:
    #         last_added_column = X.columns[i]
    #         finals.append(last_added_column)
    #         count += 1
    for f in feature_importance['feature']:
    # for f in corr_columns:
        if count < (x - len(removed)) and f not in removed:
            last_added_column = f
            finals.append(last_added_column)
            count += 1
        # print(f"{X_test_prepared_df.columns[i]:<20}"
        # f"{r.importances_mean[i]:.3f}"
        # f" +/- {r.importances_std[i]:.3f}")
    # print('finals: ',finals)
    X_train_finals = X[finals]
    # X_test_finals = X_test_prepared_df[finals]
    # X_train_finals = X_train_prepared_df.iloc[:,0:x]
    # X_test_finals = X_test_prepared_df.iloc[:,0:x]    
    error = -1 * cross_val_score(clf_best,X_train_finals.values,y,scoring='neg_mean_squared_log_error', cv=8).mean() # clf.fit(X_train_finals,y_train)
    # y_pred = clf.predict(X_test_finals)
    # error = mean_squared_log_error(y_test,y_pred)
       
    if min_error > error:
        min_error = error
    else:
        finals.remove(last_added_column)
        removed.append(last_added_column)        
        # print('removed: ',len(removed))
    print("Error: {} | Sqrt Error: {} | Features: {} | Min Error: {} | Step: {}: {}/{}".format(error,math.sqrt(error),X_train_finals.shape[1],min_error,last_added_column,x+1,X.shape[1]))