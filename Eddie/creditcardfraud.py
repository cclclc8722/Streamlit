import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import streamlit.components.v1 as components
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import sqlite3
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------------------
#Creating a login widget
with open('/Users/chengchanglei/Desktop/PycharmProjects/github/repository/RegTech_streamlit/Eddie/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


name, authentication_status, username = authenticator.login('Login', 'main')

#Authenticating users
if st.session_state["authentication_status"]:
    with st.sidebar:
        authenticator.logout('Logout', 'main')
    st.sidebar.subheader(f'*Welcome* *`{st.session_state["name"]}`* *!*')
    st.sidebar.caption(f'Login as {st.session_state["username"]}')
    uploaded_file = st.sidebar.file_uploader("Import the data here.")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data["TARGET"] = data["TARGET"].astype("O")
        st.sidebar.success("Upload data successfully!")
    else:
        st.sidebar.error("Please upload the data first.")
    task = st.sidebar.selectbox("Choose one...",
                                ["Choose one...", "Tableau expression", "Exploratory Data Analysis", "Model detective", "ML Model Analysis",'Reset password'])
    if task == "Choose one...":
        st.sidebar.warning("Please choose one.")
    elif task == "Tableau expression":
        html_temp1 = """<div class='tableauPlaceholder' id='viz1660548892258' style='position: relative'><noscript><a href='#'><img alt='儀表板窗格 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;DN&#47;DNBT3HTGS&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;DNBT3HTGS' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;DN&#47;DNBT3HTGS&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='zh-TW' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1660548892258');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp1, height=820, width=1608, scrolling=True)
    elif task == "Exploratory Data Analysis":
        st.title('Exploratory Data Analysis')
        choose = ["Choose one...", "Counting plot", "Categorical Analysis", "Bivariate Analysis", "Outlier detective"]
        choice = st.selectbox("Choose the plot that you want to show", choose)
        if task == "Choose one":
            st.text("")
        if choice == "Categorical Analysis":
            st.set_option('deprecation.showPyplotGlobalUse', False)
            defaulters = data[data.TARGET == 1]
            nondefaulters = data[data.TARGET == 0]

            def biplot(data, var, label_rotation):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 15))
                s1 = sns.barplot(ax=ax1, x=defaulters[var].value_counts().index, data=defaulters,
                                 y=100. * defaulters[var].value_counts(normalize=True))
                if (label_rotation):
                    s1.set_xticklabels(s1.get_xticklabels(), rotation=90, fontsize=12)
                ax1.set_title('Distribution of ' + '%s' % var + ' - Defaulters', fontsize=15)
                ax1.set_xlabel('%s' % var, fontsize=15)
                ax1.set_ylabel("% of Loans", fontsize=15)

                s2 = sns.barplot(ax=ax2, x=nondefaulters[var].value_counts().index, data=nondefaulters,
                                 y=100. * nondefaulters[var].value_counts(normalize=True))
                if (label_rotation):
                    s2.set_xticklabels(s2.get_xticklabels(), rotation=90, fontsize=12)
                ax2.set_xlabel('%s' % var, fontsize=15)
                ax2.set_ylabel("% of Loans", fontsize=15)
                ax2.set_title('Distribution of ' + '%s' % var + ' - Non-Defaulters', fontsize=15)
                plt.show()

            ax = st.selectbox("axis_x", data.columns)
            if data[ax].dtype == "O":
                fig = biplot(data, ax, False)
                st.pyplot(fig)
            else:
                st.warning("The data type of the data you select can not fit this plot.")
        elif choice == "Counting plot":
            ax = st.selectbox("axix_x", data.columns)
            # ay = st.selectbox("y", data.columns)
            if data[ax].dtype == "O":
                fig = plt.figure(figsize=(10, 6))
                sns.countplot(data[ax])
                st.pyplot(fig)
            else:
                st.warning("The data type of the data you select can not fit this plot.")
        elif choice == "Bivariate Analysis":
            st.set_option('deprecation.showPyplotGlobalUse', False)
            defaulters = data[data.TARGET == 1]
            nondefaulters = data[data.TARGET == 0]

            def bivariateplots(var1, var2):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 15))
                sns.scatterplot(ax=ax1, x=defaulters[var1], y=defaulters[var2], color='red')
                ax1.set_xlabel(var1)
                ax1.set_ylabel(var2)
                ax1.set_title("Defaulters")
                sns.scatterplot(ax=ax2, x=nondefaulters[var1], y=nondefaulters[var2], color='blue')
                ax2.set_xlabel(var1)
                ax2.set_ylabel(var2)
                ax2.set_title("Non Defaulters")
                plt.show()

            ax = st.selectbox("axix_x", data.columns)
            ay = st.selectbox("axix_y", data.columns)
            if data[ax].dtype == 'int64' or data[ax].dtype == 'float64' or data[ay] == 'int64' or data[
                ay].dtype == 'float64':
                fig = bivariateplots(ax, ay)
                st.pyplot(fig)
        elif choice == "Outlier detective":
            st.set_option('deprecation.showPyplotGlobalUse', False)

            def outlier_compare_charts(var, percentile):
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 15))
                s1 = sns.boxplot(ax=ax1, y=data[var], data=data, x=data['TARGET'])
                s1.set_title("With_Outliers", fontsize=10)
                s1.set_ylabel('%s' % var)

                s2 = sns.boxplot(ax=ax2, x=data['TARGET'], y=
                data[data[var] < np.nanpercentile(data[var], percentile)][var]);
                s2.set_title(var + ' boxplot on data within ' + str(percentile) + ' percentile');
                s2.set_ylabel('%s' % var)

            ax = st.selectbox("axix_x", data.columns)
            if data[ax].dtype == 'int64' or data[ax].dtype == 'float64':
                fig = outlier_compare_charts(ax, 95)
                st.pyplot(fig)
            else:
                st.warning("The data type of the data you select can not fit this plot.")

    elif task == "Model detective":
        # 標題
        st.title('Model Detective')
        # 選單
        menu = ["Choose one ...",
                "Client don't provide mobile phone number.",
                "Client's mobile phone number can't use.",
                "Client don't provide home or work phone.",
                "Client don't provide any document.",
                "The gap between income and credit limit is considerable.",
                "Client's age is under 24.",
                "Before the application did client change the identity document in 30 Days."
                ]
        indicator = st.selectbox("Choose the indicator that you want to use.", menu)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        def zero(column_name):
            column_detect = data[column_name] == 0
            tab = data.loc[column_detect, ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH',
                                           'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', column_name]]
            st.dataframe(tab)

        # Client don't provide personal phone number.
        if indicator == "Client don't provide mobile phone number.":
            zero('FLAG_MOBIL')

        # Client's personal phone number can't use.
        if indicator == "Client's mobile phone number can't use.":
            zero('FLAG_CONT_MOBILE')

        # Client don't provide home or work phone.
        if indicator == "Client don't provide home or work phone.":
            data = data.assign(for_check=data['FLAG_EMP_PHONE'] + data['FLAG_WORK_PHONE'])
            condition = data['for_check'] == 0
            table = data.loc[condition, ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH',
                                         'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']]
            st.dataframe(table)

        # Client don't provide any document.
        if indicator == "Client don't provide any document.":
            data = data.assign(Document=data['FLAG_DOCUMENT_2'] + data['FLAG_DOCUMENT_3']
                                        + data['FLAG_DOCUMENT_4'] + data['FLAG_DOCUMENT_5']
                                        + data['FLAG_DOCUMENT_6'] + data['FLAG_DOCUMENT_7']
                                        + data['FLAG_DOCUMENT_8'] + data['FLAG_DOCUMENT_9']
                                        + data['FLAG_DOCUMENT_10'] + data['FLAG_DOCUMENT_11']
                                        + data['FLAG_DOCUMENT_12'] + data['FLAG_DOCUMENT_13']
                                        + data['FLAG_DOCUMENT_14'] + data['FLAG_DOCUMENT_15']
                                        + data['FLAG_DOCUMENT_16'] + data['FLAG_DOCUMENT_17']
                                        + data['FLAG_DOCUMENT_18'] + data['FLAG_DOCUMENT_19']
                                        + data['FLAG_DOCUMENT_20'] + data['FLAG_DOCUMENT_21'])
            zero('Document')

        # The gap between income and credit limit is considerable.
        if indicator == "The gap between income and credit limit is considerable.":
            data = data.assign(Credit_Division_Income=data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL'])
            condition = data['Credit_Division_Income'] > 20
            table = data.loc[condition, ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH',
                                         'OCCUPATION_TYPE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'Credit_Division_Income']]
            st.write(table)

        # Client's age is under 24.
        if indicator == "Client's age is under 24.":
            condition = data['DAYS_BIRTH'] > (-8760)
            table = data.loc[condition, ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH',
                                         'OCCUPATION_TYPE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL']]
            st.dataframe(table)

        # Before the application did client change the identity document in 30 Days.
        if indicator == "Before the application did client change the identity document in 30 Days.":
            condition = data['DAYS_ID_PUBLISH'] > -31
            table = data.loc[condition, ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH',
                                         'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH']]
            st.dataframe(table)

    elif task == "ML Model Analysis":
        st.title("ML Analysis")
        fun = st.selectbox('Next you want to...', ['Choose one...', 'Show the dataframe', 'Show the NA analysis', 'ML Analysis'])
        if fun == "Choose one...":
            st.warning("Please choose one.")
        if fun == 'Show the dataframe':
            st.write(data.head(100))
            st.write('Shape of the dataframe: ', data.shape)
        if fun == 'Show the NA analysis':
            fig1 = plt.figure(figsize=(32, 10))
            plt.tick_params(colors='black', which='both')
            data.isna().sum()[data.isna().sum() > 0].plot(kind='bar')
            data[data['TARGET'] == 1].isna().sum()[(data[data['TARGET'] == 1].isna().sum() > 0)].plot(kind='bar', color='gray')
            plt.hlines(y=len(data), xmin=0, xmax=len(data.columns), color='red')
            plt.hlines(y=len(data[data['TARGET'] == 1]), xmin=0, xmax=len(data.columns), color='purple')
            st.pyplot(fig1)
        if fun == 'ML Analysis':
            from sklearn.preprocessing import LabelEncoder
            from sklearn.impute import SimpleImputer
            st.subheader('Establish X, Y & Label encoding')
            y = st.selectbox("Choose a variable that you would like to set as Y", ['TARGET'])
            Y = data[y].astype(str)
            X = data.drop(columns=[y, 'SK_ID_CURR'])

            X_labeled = X
            obj_list = X_labeled.select_dtypes(include="object").columns

            encoder = LabelEncoder()
            for feature in obj_list:
                X_labeled[feature] = encoder.fit_transform(X_labeled[feature].astype(str))

            imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_labeled = pd.DataFrame(imp_mean.fit_transform(X_labeled))

            if st.checkbox('Show the  labeled dataframe'):
                st.write(X_labeled.head(100))
                st.write(f'The shape of dataframe is `{X_labeled.shape}`')
                st.write(f'The shape of Y is `{Y.shape}`')
            method = st.selectbox('Choose a method...', ["Choose a method...", "Decision Tree", "Random Forest", "LGBM"])
            if method == 'Choose a method...':
                st.warning("Please choose a method you would like to use.")

            elif method == 'Decision Tree':
                from sklearn.model_selection import train_test_split
                from sklearn.utils import shuffle
                from sklearn import tree
                from sklearn import metrics
                from sklearn.metrics import plot_confusion_matrix
                import warnings
                warnings.filterwarnings('ignore')
                size = st.slider('Select the test size here', min_value=0.1, max_value=1.0, step=0.1)
                clf = st.selectbox("Choose a classifier method", ["gini", 'entropy', 'log_loss'])
                x_train, x_test, y_train, y_test = train_test_split(X_labeled, Y, test_size=size, random_state=0, shuffle=True, stratify=Y)
                clf_label = tree.DecisionTreeClassifier(criterion=clf)
                clf_label1 = clf_label.fit(x_train, y_train)

                # make prediction
                y_predict = clf_label1.predict(x_test)

                # print score
                st.set_option('deprecation.showPyplotGlobalUse', False)
                accuracy = clf_label.score(x_test, y_test)
                st.subheader(f'The accuracy rate= `{accuracy}`')
                # st.subheader(metrics.recall_score(y_test, y_predict, pos_label=0))
                # st.subheader(metrics.f1_score(y_test, y_predict, pos_label=0))

                # plot confusion matrix
                plot_confusion_matrix(clf_label, x_test, y_test, display_labels=[False, True])
                st.pyplot()

            elif method == 'Random Forest':
                # train onehot model
                from sklearn.model_selection import train_test_split
                from sklearn.utils import shuffle
                from sklearn.ensemble import RandomForestClassifier
                from sklearn import metrics
                from sklearn.metrics import plot_confusion_matrix
                st.set_option('deprecation.showPyplotGlobalUse', False)
                size = st.slider('Select the test size here', min_value=0.1, max_value=1.0, step=0.1)
                x_train, x_test, y_train, y_test = train_test_split(X_labeled, Y, test_size=size, random_state=0, shuffle=True, stratify=Y)
                clf_rndfor = RandomForestClassifier(class_weight='balanced')
                clf_rndfor = clf_rndfor.fit(x_train, y_train)

                # make prediction
                y_predict = clf_rndfor.predict(x_test)
                # print score
                accuracy = clf_rndfor.score(x_test, y_test)
                st.subheader(f'The accuracy rate= `{accuracy}`')
                # plot confusion matrix
                plot_confusion_matrix(clf_rndfor, x_test, y_test, display_labels=[False, True])
                st.pyplot()
            elif method == 'LGBM':
                # train onehot model
                from sklearn.model_selection import train_test_split
                from sklearn.utils import shuffle
                from lightgbm import LGBMClassifier
                from sklearn.metrics import plot_confusion_matrix
                from sklearn import metrics

                size = st.slider('Select the test size here', min_value=0.1, max_value=1.0, step=0.1)
                x_train, x_test, y_train, y_test = train_test_split(X_labeled, Y, test_size=size, random_state=0,shuffle=True, stratify=Y)
                clf_lgbm = LGBMClassifier()
                clf_lgbm = clf_lgbm.fit(x_train, y_train)

                # make prediction
                y_predict = clf_lgbm.predict(x_test)
                #print score
                accuracy = clf_lgbm.score(x_test, y_test)
                st.subheader(f'The accuracy rate= {accuracy}')
                # plot confusion matirx
                plot_confusion_matrix(clf_lgbm, x_test, y_test, display_labels=[False, True])
                st.pyplot()



    elif authenticator and task == 'Reset password':
        try:
            if authenticator.reset_password(username, 'Reset password'):
                st.success('Password modified successfully')
        except Exception as e:
            st.error(e)

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
    submit = st.selectbox("What do you want to do?", ["Choose one...", "Sign up", "Forgot password"])
    if submit == "Sign up":
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)
    elif submit == "Forgot password":
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password(
                'Forgot password')
            if username_forgot_pw:
                st.success('New password sent securely')
                # Random password to be transferred to user securely
            elif username_forgot_pw == False:
                st.error('Username not found')
        except Exception as e:
            st.error(e)
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')
    submit = st.selectbox("Sign up here", ["Choose one...", "Sign up", "Forgot password"])
    if submit == "Sign up":
        # Creating a new user registration widget #
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)
    elif submit == "Forgot password":
        #Creating a forgot password widget #
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
            if username_forgot_pw:
                st.success('New password sent securely')
                # Random password to be transferred to user securely #
            elif username_forgot_pw == False:
                st.error('Username not found')
        except Exception as e:
            st.error(e)



with open('/Users/chengchanglei/Desktop/PycharmProjects/github/repository/RegTech_streamlit/Eddie/config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
