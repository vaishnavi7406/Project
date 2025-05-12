import matplotlib.pyplot as plt  

days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
prices = [100, 102, 98, 105, 107]

plt.plot(days, prices, 'bo-', label='Stock Price')  # Blue line with circle markers
plt.xlabel('Days')  
plt.ylabel('Price')  
plt.title('Stock Price Trend')  
plt.legend()  # Adding legend
plt.show()
